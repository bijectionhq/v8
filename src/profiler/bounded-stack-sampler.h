// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_PROFILER_BOUNDED_STACK_SAMPLER_H_
#define V8_PROFILER_BOUNDED_STACK_SAMPLER_H_

#include <array>
#include <atomic>
#include <cstdint>

#include "include/v8-profiler.h"
#include "src/common/globals.h"

namespace v8::internal {

class Isolate;
// Isolate teardown also stops a collector whose embedder forgot its guard.
void DisposeBoundedStackSamplerForIsolate(Isolate* isolate);

// Stable slots make the weak handle locations independent of sort order.
// All fields read by the signal callback are lock-free atomics. Writers are
// serialized by Logger (or by the VM thread before the sampler starts).
// The callback never waits for a writer: an odd/changing epoch loses a sample.
class V8_EXPORT_PRIVATE BoundedNumericCodeMap {
 public:
  static constexpr uint32_t kCapacity =
      v8::BoundedStackSampler::kMaxCodeEntries;
  enum class Lookup { kFound, kUnknown, kMutation };

  void BeginMutation();
  void EndMutation();
  bool Set(uint32_t slot, Address begin, Address end, int script_id,
           int function_start_utf16);
  void Clear(uint32_t slot);
  Address begin(uint32_t slot) const;
  Address end(uint32_t slot) const;
  Lookup LookupFrame(Address pc, v8::BoundedStackSampler::Frame* out) const;

 private:
  struct Entry {
    std::atomic<Address> begin{0};
    std::atomic<Address> end{0};
    std::atomic<int32_t> script_id{-1};
    std::atomic<int32_t> function_start_utf16{-1};
  };
  static_assert(std::atomic<Address>::is_always_lock_free);
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  static_assert(std::atomic<int32_t>::is_always_lock_free);
  std::atomic<uint64_t> epoch_{0};
  std::atomic<uint32_t> count_{0};
  std::array<Entry, kCapacity> entries_{};
  std::array<std::atomic<uint32_t>, kCapacity> sorted_{};
  std::array<uint32_t, kCapacity> scratch_{};
};

}  // namespace v8::internal

#endif  // V8_PROFILER_BOUNDED_STACK_SAMPLER_H_
