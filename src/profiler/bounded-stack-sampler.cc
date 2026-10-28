// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/profiler/bounded-stack-sampler.h"

#include <algorithm>
#include <limits>
#include <new>

#include "src/api/api-inl.h"
#include "src/base/platform/condition-variable.h"
#include "src/base/platform/platform.h"
#include "src/execution/isolate-inl.h"
#include "src/execution/v8threads.h"
#include "src/handles/global-handles-inl.h"
#include "src/heap/heap.h"
#include "src/libsampler/sampler.h"
#include "src/logging/code-events.h"
#include "src/objects/bytecode-array-inl.h"
#include "src/objects/code-inl.h"
#include "src/objects/js-function-inl.h"
#include "src/objects/script-inl.h"
#include "src/objects/shared-function-info-inl.h"
#include "src/profiler/tick-sample.h"
#include "src/sandbox/hardware-support.h"

#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
namespace v8::internal {

void BoundedNumericCodeMap::BeginMutation() {
  uint64_t previous = epoch_.fetch_add(1, std::memory_order_acq_rel);
  DCHECK_EQ(previous & 1, 0);
}

void BoundedNumericCodeMap::EndMutation() {
  uint32_t count = 0;
  for (uint32_t i = 0; i < kCapacity; ++i) {
    if (begin(i) != 0) scratch_[count++] = i;
  }
  std::sort(scratch_.begin(), scratch_.begin() + count,
            [this](uint32_t a, uint32_t b) { return begin(a) < begin(b); });
  for (uint32_t i = 0; i < count; ++i) {
    sorted_[i].store(scratch_[i], std::memory_order_relaxed);
  }
  count_.store(count, std::memory_order_relaxed);
  epoch_.fetch_add(1, std::memory_order_release);
}

bool BoundedNumericCodeMap::Set(uint32_t slot, Address begin, Address end,
                                int script_id, int function_start_utf16) {
  if (slot >= kCapacity || begin == 0 || end <= begin || script_id < 0 ||
      function_start_utf16 < 0) {
    return false;
  }
  entries_[slot].begin.store(begin, std::memory_order_relaxed);
  entries_[slot].end.store(end, std::memory_order_relaxed);
  entries_[slot].script_id.store(script_id, std::memory_order_relaxed);
  entries_[slot].function_start_utf16.store(function_start_utf16,
                                           std::memory_order_relaxed);
  return true;
}

void BoundedNumericCodeMap::Clear(uint32_t slot) {
  DCHECK_LT(slot, kCapacity);
  entries_[slot].begin.store(0, std::memory_order_relaxed);
}

Address BoundedNumericCodeMap::begin(uint32_t slot) const {
  return entries_[slot].begin.load(std::memory_order_relaxed);
}

Address BoundedNumericCodeMap::end(uint32_t slot) const {
  return entries_[slot].end.load(std::memory_order_relaxed);
}

BoundedNumericCodeMap::Lookup BoundedNumericCodeMap::LookupFrame(
    Address pc, v8::BoundedStackSampler::Frame* out) const {
  const uint64_t epoch = epoch_.load(std::memory_order_acquire);
  if (epoch & 1) return Lookup::kMutation;
  uint32_t lo = 0;
  uint32_t hi = count_.load(std::memory_order_relaxed);
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    const uint32_t slot = sorted_[mid].load(std::memory_order_relaxed);
    if (begin(slot) <= pc) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  Lookup result = Lookup::kUnknown;
  v8::BoundedStackSampler::Frame frame;
  if (lo > 0) {
    const uint32_t slot = sorted_[lo - 1].load(std::memory_order_relaxed);
    if (begin(slot) != 0 && pc < end(slot)) {
      frame.script_id = entries_[slot].script_id.load(std::memory_order_relaxed);
      frame.function_start_utf16 =
          entries_[slot].function_start_utf16.load(std::memory_order_relaxed);
      result = Lookup::kFound;
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  if (epoch_.load(std::memory_order_relaxed) != epoch) {
    return Lookup::kMutation;
  }
  *out = frame;
  return result;
}

namespace {
using Api = v8::BoundedStackSampler;
// Admission includes stopped-but-undisposed arenas, not just active threads.
std::array<std::atomic<Isolate*>, 4> owners{};
std::array<std::atomic<void*>, 4> collectors{};

class BoundedStackSamplerImpl final : public LogEventListener {
 public:
  static BoundedStackSamplerImpl* New(Isolate* isolate,
                                      Tagged<NativeContext> context,
                                      const Api::Options& options) {
    if (options.sampling_interval_us < 1000 ||
        options.sampling_interval_us > 10000 || options.max_samples == 0 ||
        options.max_samples > Api::kMaxSamples ||
        options.max_code_entries == 0 ||
        options.max_code_entries > Api::kMaxCodeEntries ||
        options.max_rundown_objects == 0 ||
        options.max_rundown_objects > Api::kMaxRundownObjects) {
      return nullptr;
    }
    for (auto& owner : owners) {
      if (owner.load(std::memory_order_acquire) == isolate) return nullptr;
    }
    uint32_t admission = 0;
    for (; admission < owners.size(); ++admission) {
      Isolate* empty = nullptr;
      if (owners[admission].compare_exchange_strong(
              empty, isolate, std::memory_order_acq_rel)) {
        break;
      }
    }
    if (admission == owners.size()) return nullptr;
    auto* result = new (std::nothrow) BoundedStackSamplerImpl(
        isolate, context, options, admission);
    if (result == nullptr) {
      owners[admission].store(nullptr, std::memory_order_release);
      return nullptr;
    }
    collectors[admission].store(result, std::memory_order_release);
    if (!result->Start()) {
      delete result;
      return nullptr;
    }
    return result;
  }

  ~BoundedStackSamplerImpl() override {
    Stop();
    collectors[admission_].store(nullptr, std::memory_order_release);
    owners[admission_].store(nullptr, std::memory_order_release);
  }

  void Stop() {
    DCHECK_EQ(ThreadId::Current(), owner_thread_);
    if (stopped_) return;
    stop_requested_.store(true, std::memory_order_release);
    if (timer_started_) {
      timer_.Wake();
      timer_.Join();
      signal_sampler_.Stop();
    }
    if (attached_) {
      CHECK(isolate_->logger()->RemoveListener(this));
      isolate_->UpdateLogObjectRelocation();
    }
    for (auto& weak : weak_code_) {
      if (weak.location != nullptr) GlobalHandles::Destroy(weak.location);
      weak.location = nullptr;
    }
    if (context_location_ != nullptr) GlobalHandles::Destroy(context_location_);
    context_location_ = nullptr;
    summary_.sample_count = count_.load(std::memory_order_relaxed);
    summary_.loss = loss_.load(std::memory_order_relaxed);
    summary_.non_js_observations = non_js_.load(std::memory_order_relaxed);
    summary_.other_context_observations = other_context_.load(std::memory_order_relaxed);
    summary_.map_mutation_observations = mutations_.load(std::memory_order_relaxed);
    summary_.unavailable_stack_observations = unavailable_.load(std::memory_order_relaxed);
    summary_.elapsed_us = Offset();
    const int64_t cutoff = cutoff_us_.load(std::memory_order_relaxed);
    summary_.collection_end_us = cutoff < 0 ? summary_.elapsed_us : cutoff;
    stopped_ = true;
  }

  Api::Summary summary() const {
    CHECK(stopped_);
    return summary_;
  }
  bool sample(uint32_t index, Api::Sample* out) const {
    CHECK(stopped_);
    if (index >= summary_.sample_count || out == nullptr) return false;
    *out = samples_[index];
    return true;
  }

  bool is_listening_to_code_events() override { return true; }

  void CodeCreateEvent(CodeTag, DirectHandle<AbstractCode> code,
                       const char*) override {
    RegisterCode(code, {});
  }
  void CodeCreateEvent(CodeTag, DirectHandle<AbstractCode> code,
                       DirectHandle<Name>) override {
    RegisterCode(code, {});
  }
  void CodeCreateEvent(CodeTag, DirectHandle<AbstractCode> code,
                       DirectHandle<SharedFunctionInfo> shared,
                       DirectHandle<Name>) override {
    RegisterCode(code, shared);
  }
  void CodeCreateEvent(CodeTag, DirectHandle<AbstractCode> code,
                       DirectHandle<SharedFunctionInfo> shared,
                       DirectHandle<Name>, int, int) override {
    RegisterCode(code, shared);
  }
#if V8_ENABLE_WEBASSEMBLY
  void CodeCreateEvent(CodeTag, const wasm::WasmCode*, wasm::WasmName,
                       std::string_view, int, int) override {}
#endif
  void CallbackEvent(DirectHandle<Name>, Address) override {}
  void GetterCallbackEvent(DirectHandle<Name>, Address) override {}
  void SetterCallbackEvent(DirectHandle<Name>, Address) override {}
  void RegExpCodeCreateEvent(DirectHandle<AbstractCode> code,
                             DirectHandle<String>, regexp::Flags) override {
    RegisterCode(code, {});
  }
  void CodeMoveEvent(Tagged<InstructionStream> from,
                     Tagged<InstructionStream> to) override {
    MoveCode(from->instruction_start(), to->instruction_start(), to->body_size());
  }
  void BytecodeMoveEvent(Tagged<BytecodeArray> from,
                         Tagged<BytecodeArray> to) override {
    MoveCode(from->GetFirstBytecodeAddress(), to->GetFirstBytecodeAddress(), to->length());
  }
  void NativeContextMoveEvent(Address from, Address to) override {
    Address expected = from;
    context_.compare_exchange_strong(expected, to, std::memory_order_relaxed);
  }
  void SharedFunctionInfoMoveEvent(Address, Address) override {}
  void CodeMovingGCEvent() override {}
  void CodeDisableOptEvent(DirectHandle<AbstractCode>,
                           DirectHandle<SharedFunctionInfo>) override {}
  void CodeDeoptEvent(DirectHandle<Code>, DeoptimizeKind, Address, int) override {}
  void CodeDependencyChangeEvent(DirectHandle<Code>,
                                 DirectHandle<SharedFunctionInfo>,
                                 const char*) override {}
  void WeakCodeClearEvent() override {
    // MarkCompactCollector emits this on the VM thread after joining parallel
    // clearing jobs. The weak locations have been nulled before address reuse.
    DCHECK_EQ(ThreadId::Current(), owner_thread_);
    // Do not keep an obsolete request context alive. Clearing the numeric
    // address before execution resumes also prevents a later context at the
    // same address from being mistaken for this request.
    if (context_location_ == nullptr) {
      context_.store(0, std::memory_order_release);
      Freeze(Api::kContextCollected);
    }
    map_.BeginMutation();
    for (uint32_t i = 0; i < options_.max_code_entries; ++i) {
      if (weak_code_[i].location == nullptr) map_.Clear(i);
    }
    ReclaimRetired();
    map_.EndMutation();
  }

 private:
  class SignalSampler final : public sampler::Sampler {
   public:
    explicit SignalSampler(BoundedStackSamplerImpl* owner)
        : Sampler(reinterpret_cast<v8::Isolate*>(owner->isolate_)),
          owner_(owner) {}
    void SampleStack(const v8::RegisterState& regs) override {
      owner_->Record(regs);
    }
   private:
    BoundedStackSamplerImpl* const owner_;
  };
  class Timer final : public base::Thread {
   public:
    explicit Timer(BoundedStackSamplerImpl* owner)
        : Thread(Options("v8:BoundedSamp", 256 * KB)), owner_(owner) {}
    void Wake() { condition_.NotifyOne(); }
    void Run() override {
      SetCurrentIsolateScope isolate_scope(owner_->isolate_);
      base::MutexGuard guard(&mutex_);
      while (!owner_->stop_requested_.load(std::memory_order_acquire)) {
        condition_.WaitFor(&mutex_, base::TimeDelta::FromMicroseconds(
                                       owner_->options_.sampling_interval_us));
        if (owner_->stop_requested_.load(std::memory_order_acquire)) break;
        if (owner_->Offset() >= Api::kMaxCollectionDurationUs) {
          owner_->Freeze(Api::kTimeLimit);
          break;
        }
        owner_->signal_sampler_.DoSample();
      }
    }
   private:
    BoundedStackSamplerImpl* const owner_;
    base::Mutex mutex_;
    base::ConditionVariable condition_;
  };
  struct WeakCode {
    // Stable address passed to GlobalHandles::MakeWeak, as in WeakCodeRegistry.
    Address* location = nullptr;
    bool retired = false;
  };

  BoundedStackSamplerImpl(Isolate* isolate, Tagged<NativeContext> context,
                          const Api::Options& options, uint32_t admission)
      : isolate_(isolate), options_(options), admission_(admission),
        owner_thread_(ThreadId::Current()),
        thread_data_(isolate->FindPerThreadDataForThisThread()),
        context_(context.address()), start_(base::TimeTicks::Now()),
        signal_sampler_(this), timer_(this) {
    context_location_ = isolate_->global_handles()->Create(context).location();
    GlobalHandles::MakeWeak(&context_location_);
  }

  bool Start() {
    CHECK(isolate_->logger()->AddListener(this));
    attached_ = true;
    // Unlike SetIsProfiling(true), this does not request source positions.
    isolate_->UpdateLogObjectRelocation();
    Rundown();
    if (stop_requested_.load(std::memory_order_acquire)) return true;
    signal_sampler_.Start();
    if (!timer_.StartSynchronously()) {
      signal_sampler_.Stop();
      return false;
    }
    timer_started_ = true;
    return true;
  }

  int64_t Offset() const {
    return (base::TimeTicks::Now() - start_).InMicroseconds();
  }
  void Freeze(uint32_t reason) {
    loss_.fetch_or(reason, std::memory_order_relaxed);
    int64_t unset = -1;
    cutoff_us_.compare_exchange_strong(unset, Offset(), std::memory_order_relaxed);
    stop_requested_.store(true, std::memory_order_release);
  }

  void ReclaimRetired() {
    // Never destroy global handles on a parallel GC migration worker.
    DCHECK_EQ(ThreadId::Current(), owner_thread_);
    for (auto& weak : weak_code_) {
      if (weak.retired) {
        if (weak.location != nullptr) GlobalHandles::Destroy(weak.location);
        weak.location = nullptr;
        weak.retired = false;
      }
    }
  }

  void InvalidateOverlaps(Address begin, Address end, uint32_t except) {
    for (uint32_t i = 0; i < options_.max_code_entries; ++i) {
      if (i != except && map_.begin(i) != 0 && map_.begin(i) < end &&
          map_.end(i) > begin) {
        map_.Clear(i);
        weak_code_[i].retired = true;
      }
    }
  }

  void RegisterCode(DirectHandle<AbstractCode> code,
                     DirectHandle<SharedFunctionInfo> shared) {
    if (stop_requested_.load(std::memory_order_acquire)) return;
    if (Offset() >= Api::kMaxCollectionDurationUs) {
      Freeze(Api::kTimeLimit);
      return;
    }
    if (ThreadId::Current() != owner_thread_) {
      Freeze(Api::kConcurrentCodeCreation);
      return;
    }
    if (++code_events_ > Api::kMaxCodeEvents) {
      Freeze(Api::kCodeEventLimit);
      return;
    }
    DisallowGarbageCollection no_gc;
    PtrComprCageBase cage_base(isolate_);
    const Address begin = code->InstructionStart(cage_base);
    const unsigned size = code->InstructionSize(cage_base);
    if (size == 0 || begin > std::numeric_limits<Address>::max() - size) return;
    if (!batching_) map_.BeginMutation();
    InvalidateOverlaps(begin, begin + size, Api::kMaxCodeEntries);
    ReclaimRetired();
    int script_id = -1;
    int start = -1;
    if (!shared.is_null()) {
      Tagged<Object> script = shared->raw_script(kAcquireLoad);
      if (IsScript(script)) {
        script_id = Cast<Script>(script)->id();
        start = shared->StartPosition();
      }
    }
    if (script_id >= 0 && start >= 0) {
      uint32_t slot = 0;
      for (; slot < options_.max_code_entries; ++slot) {
        if (weak_code_[slot].location == nullptr) break;
      }
      if (slot == options_.max_code_entries) {
        Freeze(Api::kCodeLimit);
      } else {
        Handle<AbstractCode> handle = isolate_->global_handles()->Create(*code);
        weak_code_[slot].location = handle.location();
        GlobalHandles::MakeWeak(&weak_code_[slot].location);
        identities_[slot] = {script_id, start};
        CHECK(map_.Set(slot, begin, begin + size, script_id, start));
      }
    }
    if (!batching_) map_.EndMutation();
  }

  void MoveCode(Address from, Address to, Address moved_size) {
    if (stop_requested_.load(std::memory_order_acquire)) return;
    if (from == to) return;
    // Logger serializes migration callbacks, including parallel GC workers.
    // Signal readers do not take that mutex and never access weak handles.
    map_.BeginMutation();
    uint32_t slot = 0;
    for (; slot < options_.max_code_entries; ++slot) {
      if (map_.begin(slot) == from) break;
    }
    if (slot < options_.max_code_entries) {
      const Address size = map_.end(slot) - from;
      if (to <= std::numeric_limits<Address>::max() - size) {
        Api::Frame frame;
        // The map is deliberately odd. The separate stable numeric identity
        // is only accessed by Logger-serialized writers, never by signals.
        frame = identities_[slot];
        bool conflict = false;
        for (uint32_t i = 0; i < options_.max_code_entries; ++i) {
          if (i != slot && map_.begin(i) != 0 && map_.begin(i) < to + size &&
              map_.end(i) > to) conflict = true;
        }
        // Never guess which identity owns a colliding migration destination.
        // Keep already resolved samples and stop before any post-GC sample.
        if (conflict) {
          Freeze(Api::kRelocationConflict);
        } else {
          CHECK(map_.Set(slot, to, to + size, frame.script_id,
                          frame.function_start_utf16));
        }
      } else {
        Freeze(Api::kCodeLimit);
      }
    } else if (to <= std::numeric_limits<Address>::max() - moved_size) {
      for (uint32_t i = 0; i < options_.max_code_entries; ++i) {
        if (map_.begin(i) != 0 && map_.begin(i) < to + moved_size &&
            map_.end(i) > to) {
          Freeze(Api::kRelocationConflict);
          break;
        }
      }
    }
    map_.EndMutation();
  }

  void Rundown() {
    // Stream primitive identities while GC is disallowed. Do not call the
    // stock LogCompiledFunctions: it builds growable handles/set/vector state
    // and materializes source positions before invoking listeners.
    HandleScope handles(isolate_);
    batching_ = true;
    map_.BeginMutation();
    uint32_t visited = 0;
    auto visit = [&]() {
      if (Offset() >= Api::kMaxCollectionDurationUs) {
        Freeze(Api::kTimeLimit);
        return false;
      }
      if (++visited > options_.max_rundown_objects) {
        Freeze(Api::kRundownLimit);
        return false;
      }
      return !stop_requested_.load(std::memory_order_relaxed);
    };
    {
      HeapObjectIterator iterator(isolate_->heap());
      for (Tagged<HeapObject> object = iterator.Next(); !object.is_null();
           object = iterator.Next()) {
        if (!visit()) break;
        HandleScope code_handles(isolate_);
        if (IsJSFunction(object)) {
          Tagged<JSFunction> function = Cast<JSFunction>(object);
          if (!function->shared()->HasUnpublishedTrustedData(isolate_) &&
              function->HasAttachedOptimizedCode(isolate_)) {
            RegisterCode(direct_handle(Cast<AbstractCode>(function->code(isolate_)), isolate_),
                         direct_handle(function->shared(), isolate_));
          }
        }
      }
      Script::Iterator scripts(isolate_);
      for (Tagged<Script> script = scripts.Next(); !script.is_null();
           script = scripts.Next()) {
        if (!visit()) break;
        if (!script->HasValidSource()) continue;
        HandleScope script_handles(isolate_);
        SharedFunctionInfo::ScriptIterator functions(isolate_, script);
        for (Tagged<SharedFunctionInfo> shared = functions.Next();
             !shared.is_null(); shared = functions.Next()) {
          if (!visit()) break;
          HandleScope function_handles(isolate_);
          if (shared->HasUnpublishedTrustedData(isolate_) || !shared->is_compiled()) continue;
          RegisterCode(direct_handle(Cast<AbstractCode>(shared->abstract_code(isolate_)), isolate_),
                       direct_handle(shared, isolate_));
          if (shared->HasBaselineCode()) {
            RegisterCode(direct_handle(Cast<AbstractCode>(
                                           shared->baseline_code(kAcquireLoad)),
                                       isolate_),
                         direct_handle(shared, isolate_));
          }
        }
      }
    }
    batching_ = false;
    map_.EndMutation();
  }

  void Record(const v8::RegisterState& state) {
    if (stop_requested_.load(std::memory_order_acquire)) return;
    if (isolate_->was_locker_ever_used() &&
        (!isolate_->thread_manager()->IsLockedByThread(thread_data_->thread_id()) ||
         thread_data_->thread_state() != nullptr)) {
      unavailable_.fetch_add(1, std::memory_order_relaxed);
      loss_.fetch_or(Api::kStackUnavailable, std::memory_order_relaxed);
      return;
    }
    void* pcs[Api::kMaxFrames + 1];
    // RegisterState's copy constructor may allocate callee_saved storage.
    // GetStackSample on the supported native architectures reads these four
    // registers only; leave callee_saved empty in the signal handler.
    v8::RegisterState registers;
    registers.pc = state.pc;
    registers.sp = state.sp;
    registers.fp = state.fp;
    registers.lr = state.lr;
    v8::SampleInfo info;
    StateTag actual_state = OTHER;
    if (!TickSample::GetStackSample(isolate_, &registers,
                                    TickSample::kSkipCEntryFrame, pcs,
                                    Api::kMaxFrames + 1, &info, &actual_state)) {
      unavailable_.fetch_add(1, std::memory_order_relaxed);
      loss_.fetch_or(Api::kStackUnavailable, std::memory_order_relaxed);
      return;
    }
    if (!IsExternal(actual_state)) actual_state = info.vm_state;
    if (actual_state != JS) {
      non_js_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const Address sampled_context = reinterpret_cast<Address>(info.context) & ~kHeapObjectTag;
    if (sampled_context != context_.load(std::memory_order_relaxed)) {
      other_context_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const uint32_t index = count_.load(std::memory_order_relaxed);
    Api::Sample& sample = samples_[index];
    sample = Api::Sample{};
    sample.offset_us = Offset();
    // As in SymbolizeTickSample, the interrupted PC precedes the caller PCs
    // returned by GetStackSample. Omitting it loses optimized leaf functions.
    // In the interpreter it is generally an unmapped VM bytecode handler;
    // the following bytecode-array frame identifies the JS activation.
    const uint32_t frames = 1 + static_cast<uint32_t>(info.frames_count);
    sample.frame_count = std::min<uint32_t>(frames, Api::kMaxFrames);
    sample.loss = frames > Api::kMaxFrames ? Api::kDepthLimit : 0;
    for (uint32_t i = 0; i < sample.frame_count; ++i) {
      const Address pc = reinterpret_cast<Address>(i == 0 ? registers.pc : pcs[i - 1]);
      const auto result = map_.LookupFrame(pc, &sample.frames[i]);
      if (result == BoundedNumericCodeMap::Lookup::kMutation) {
        mutations_.fetch_add(1, std::memory_order_relaxed);
        loss_.fetch_or(Api::kInterruptedMapMutation, std::memory_order_relaxed);
        return;
      }
      if (result == BoundedNumericCodeMap::Lookup::kUnknown) sample.loss |= Api::kUnmappedFrame;
    }
    if (stop_requested_.load(std::memory_order_acquire)) return;
    loss_.fetch_or(sample.loss, std::memory_order_relaxed);
    count_.store(index + 1, std::memory_order_release);
    if (index + 1 == options_.max_samples) Freeze(Api::kSampleLimit);
  }

  Isolate* const isolate_;
  const Api::Options options_;
  const uint32_t admission_;
  const ThreadId owner_thread_;
  Isolate::PerIsolateThreadData* const thread_data_;
  Address* context_location_ = nullptr;
  std::atomic<Address> context_;
  const base::TimeTicks start_;
  BoundedNumericCodeMap map_;
  std::array<WeakCode, Api::kMaxCodeEntries> weak_code_{};
  std::array<Api::Frame, Api::kMaxCodeEntries> identities_{};
  std::array<Api::Sample, Api::kMaxSamples> samples_{};
  std::atomic<uint32_t> count_{0}, loss_{0}, non_js_{0}, other_context_{0},
      mutations_{0}, unavailable_{0};
  std::atomic<int64_t> cutoff_us_{-1};
  std::atomic<bool> stop_requested_{false};
  bool batching_ = false, attached_ = false, timer_started_ = false, stopped_ = false;
  uint32_t code_events_ = 0;
  Api::Summary summary_;
  SignalSampler signal_sampler_;
  Timer timer_;
};

static_assert(sizeof(BoundedStackSamplerImpl) <= 512 * KB);
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<int64_t>::is_always_lock_free);
static_assert(std::atomic<bool>::is_always_lock_free);

}  // namespace

void DisposeBoundedStackSamplerForIsolate(Isolate* isolate) {
  for (uint32_t i = 0; i < owners.size(); ++i) {
    if (owners[i].load(std::memory_order_acquire) == isolate) {
      delete static_cast<BoundedStackSamplerImpl*>(
          collectors[i].load(std::memory_order_acquire));
      return;
    }
  }
}
}  // namespace v8::internal
#else
namespace v8::internal {
void DisposeBoundedStackSamplerForIsolate(Isolate*) {}
}  // namespace v8::internal
#endif

namespace v8 {
BoundedStackSampler* BoundedStackSampler::New(Isolate* isolate,
                                              Local<Context> context,
                                              const Options& options) {
#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
  auto* internal_isolate = reinterpret_cast<internal::Isolate*>(isolate);
  if (context.IsEmpty() || context->GetIsolate() != isolate ||
      Isolate::TryGetCurrent() != isolate || !isolate->IsInUse() ||
      !internal::HardwareSandboxingDisabledOrSupportsSignalDeliveryInSandbox()) return nullptr;
  auto native_context = Utils::OpenDirectHandle(*context)->native_context();
  return reinterpret_cast<BoundedStackSampler*>(
      internal::BoundedStackSamplerImpl::New(internal_isolate, native_context, options));
#else
  return nullptr;
#endif
}
void BoundedStackSampler::Stop() {
#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
  reinterpret_cast<internal::BoundedStackSamplerImpl*>(this)->Stop();
#else
  UNREACHABLE();
#endif
}
BoundedStackSampler::Summary BoundedStackSampler::GetSummary() const {
#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
  return reinterpret_cast<const internal::BoundedStackSamplerImpl*>(this)->summary();
#else
  UNREACHABLE();
#endif
}
bool BoundedStackSampler::GetSample(uint32_t index, Sample* out) const {
#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
  return reinterpret_cast<const internal::BoundedStackSamplerImpl*>(this)->sample(index, out);
#else
  UNREACHABLE();
#endif
}
void BoundedStackSampler::Dispose() {
#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
  delete reinterpret_cast<internal::BoundedStackSamplerImpl*>(this);
#else
  UNREACHABLE();
#endif
}
}  // namespace v8
