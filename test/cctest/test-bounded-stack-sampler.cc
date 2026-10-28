// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <memory>

#include "include/v8-profiler.h"
#include "src/profiler/bounded-stack-sampler.h"
#include "src/sandbox/hardware-support.h"
#include "test/cctest/cctest.h"

#if (V8_OS_LINUX || V8_OS_DARWIN) && (V8_HOST_ARCH_X64 || V8_HOST_ARCH_ARM64)
namespace v8::internal {
namespace {
using Api = v8::BoundedStackSampler;

TEST(BoundedStackMapPublicationAndAddressReuse) {
  auto map = std::make_unique<BoundedNumericCodeMap>();
  Api::Frame frame;
  map->BeginMutation();
  CHECK(map->Set(0, 100, 200, 17, 23));
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kMutation,
           map->LookupFrame(150, &frame));
  map->EndMutation();
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kFound,
           map->LookupFrame(150, &frame));
  CHECK_EQ(17, frame.script_id);
  CHECK_EQ(23, frame.function_start_utf16);
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kUnknown,
           map->LookupFrame(200, &frame));

  // A cleared weak code slot must not attribute an address reused by a new
  // code object to the old script. Moving the new object changes its range.
  map->BeginMutation();
  map->Clear(0);
  CHECK(map->Set(1, 100, 180, 29, 31));
  map->EndMutation();
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kFound,
           map->LookupFrame(150, &frame));
  CHECK_EQ(29, frame.script_id);
  map->BeginMutation();
  CHECK(map->Set(1, 300, 380, 29, 31));
  map->EndMutation();
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kUnknown,
           map->LookupFrame(150, &frame));
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kFound,
           map->LookupFrame(350, &frame));
  CHECK_EQ(29, frame.script_id);
}

TEST(BoundedStackMapFixedCapacity) {
  auto map = std::make_unique<BoundedNumericCodeMap>();
  map->BeginMutation();
  for (uint32_t i = 0; i < BoundedNumericCodeMap::kCapacity; ++i) {
    CHECK(map->Set(i, 100 + i * 16, 108 + i * 16, i, i * 2));
  }
  CHECK(!map->Set(BoundedNumericCodeMap::kCapacity, 1, 2, 1, 0));
  CHECK(!map->Set(0, 1, 2, -1, 0));
  CHECK(!map->Set(0, 1, 1, 1, 0));
  map->EndMutation();
  Api::Frame frame;
  CHECK_EQ(BoundedNumericCodeMap::Lookup::kFound,
           map->LookupFrame(100 + 4095 * 16, &frame));
  CHECK_EQ(4095, frame.script_id);
}

v8::Local<v8::Script> Compile(v8::Isolate* isolate,
                             v8::Local<v8::Context> context,
                             const char* source) {
  return v8::Script::Compile(
             context, v8::String::NewFromUtf8(isolate, source).ToLocalChecked())
      .ToLocalChecked();
}

void Run(v8::Isolate* isolate, v8::Local<v8::Context> context,
          const char* source) {
  Compile(isolate, context, source)->Run(context).ToLocalChecked();
}

TEST(BoundedStackSamplerKnownFramesSurviveGcAndContextReuse) {
  if (!HardwareSandboxingDisabledOrSupportsSignalDeliveryInSandbox()) return;
  CcTest::InitializeVM();
  auto allocator = std::unique_ptr<v8::ArrayBuffer::Allocator>(
      v8::ArrayBuffer::Allocator::NewDefaultAllocator());
  v8::Isolate::CreateParams params;
  params.array_buffer_allocator = allocator.get();
  auto* isolate = v8::Isolate::New(params);
  {
    v8::Isolate::Scope entered(isolate);
    v8::HandleScope handles(isolate);
    LocalContext context(isolate);
    auto first = Compile(isolate, context.local(),
        "function first(ms) { let n=0; const end=Date.now()+ms;"
        "while (Date.now()<end) { for(let i=0;i<10000;i++) n=Math.imul(n+1,3); } return n; }");
    auto second = Compile(isolate, context.local(),
        "function second(ms) { let n=0; const end=Date.now()+ms;"
        "while (Date.now()<end) { for(let i=0;i<10000;i++) n=Math.imul(n+1,5); } return n; }");
    const int first_id = first->GetUnboundScript()->GetId();
    const int second_id = second->GetUnboundScript()->GetId();
    first->Run(context.local()).ToLocalChecked();
    second->Run(context.local()).ToLocalChecked();
    Run(isolate, context.local(), "first(2);second(2)");
    Api::Options options;
    auto* sampler = Api::New(isolate, context.local(), options);
    CHECK_NOT_NULL(sampler);
    CHECK_NULL(Api::New(isolate, context.local(), options));
    Run(isolate, context.local(), "first(40)");
    isolate->LowMemoryNotification();
    Run(isolate, context.local(), "second(40)");
    sampler->Stop();
    const Api::Summary summary = sampler->GetSummary();
    CHECK_GT(summary.sample_count, 0);
    CHECK_LE(summary.sample_count, Api::kMaxSamples);
    bool saw_first = false, saw_second = false;
    for (uint32_t i = 0; i < summary.sample_count; ++i) {
      Api::Sample sample;
      CHECK(sampler->GetSample(i, &sample));
      CHECK_LE(sample.frame_count, Api::kMaxFrames);
      CHECK_GE(sample.offset_us, 0);
      CHECK_LE(sample.offset_us, summary.elapsed_us);
      for (uint32_t j = 0; j < sample.frame_count; ++j) {
        saw_first |= sample.frames[j].script_id == first_id;
        saw_second |= sample.frames[j].script_id == second_id;
        if (sample.frames[j].script_id >= 0) {
          CHECK_GE(sample.frames[j].function_start_utf16, 0);
        }
      }
    }
    // This is a real sampler test: empty or wholly unmapped samples fail it.
    CHECK(saw_first);
    CHECK(saw_second);
    sampler->Dispose();

    options.max_samples = 1;
    sampler = Api::New(isolate, context.local(), options);
    CHECK_NOT_NULL(sampler);
    Run(isolate, context.local(), "second(40)");
    sampler->Stop();
    CHECK_EQ(1, sampler->GetSummary().sample_count);
    CHECK_NE(0, sampler->GetSummary().loss & Api::kSampleLimit);
    sampler->Dispose();
  }
  isolate->Dispose();
}

TEST(BoundedStackSamplerInvalidBoundsAndForgottenCollectorCleanup) {
  if (!HardwareSandboxingDisabledOrSupportsSignalDeliveryInSandbox()) return;
  CcTest::InitializeVM();
  auto allocator = std::unique_ptr<v8::ArrayBuffer::Allocator>(
      v8::ArrayBuffer::Allocator::NewDefaultAllocator());
  // More than four successive isolates proves teardown releases admission;
  // their sampler threads must also be joined before their isolates disappear.
  for (int i = 0; i < 5; ++i) {
    v8::Isolate::CreateParams params;
    params.array_buffer_allocator = allocator.get();
    auto* isolate = v8::Isolate::New(params);
    {
      v8::Isolate::Scope entered(isolate);
      v8::HandleScope handles(isolate);
      LocalContext context(isolate);
      Api::Options options;
      options.max_samples = Api::kMaxSamples + 1;
      CHECK_NULL(Api::New(isolate, context.local(), options));
      Run(isolate, context.local(),
          "function a(){return 1} function b(){return 2} a(); b();");
      options.max_samples = 1;
      options.max_code_entries = 1;
      auto* limited = Api::New(isolate, context.local(), options);
      CHECK_NOT_NULL(limited);
      limited->Stop();
      CHECK_NE(0, limited->GetSummary().loss & Api::kCodeLimit);
      limited->Dispose();
      options.max_code_entries = Api::kMaxCodeEntries;
      options.max_samples = 1;
      CHECK_NOT_NULL(Api::New(isolate, context.local(), options));
      // Intentionally no Dispose: Isolate::Deinit owns forgotten collectors.
    }
    isolate->Dispose();
  }
}

TEST(BoundedStackSamplerDoesNotRetainObsoleteContext) {
  if (!HardwareSandboxingDisabledOrSupportsSignalDeliveryInSandbox()) return;
  CcTest::InitializeVM();
  auto allocator = std::unique_ptr<v8::ArrayBuffer::Allocator>(
      v8::ArrayBuffer::Allocator::NewDefaultAllocator());
  v8::Isolate::CreateParams params;
  params.array_buffer_allocator = allocator.get();
  auto* isolate = v8::Isolate::New(params);
  {
    v8::Isolate::Scope entered(isolate);
    Api* sampler = nullptr;
    v8::Global<v8::Context> weak;
    {
      v8::HandleScope handles(isolate);
      auto context = v8::Context::New(isolate);
      v8::Context::Scope context_scope(context);
      weak.Reset(isolate, context);
      weak.SetWeak();
      sampler = Api::New(isolate, context, Api::Options{});
      CHECK_NOT_NULL(sampler);
    }
    isolate->LowMemoryNotification();
    CHECK(weak.IsEmpty());
    // Reusing/creating a context after GC cannot reactivate the old identity.
    {
      v8::HandleScope handles(isolate);
      auto replacement = v8::Context::New(isolate);
      v8::Context::Scope replacement_scope(replacement);
      Run(isolate, replacement, "let n=0;for(let i=0;i<100000;i++)n+=i;");
    }
    sampler->Stop();
    CHECK_NE(0, sampler->GetSummary().loss & Api::kContextCollected);
    CHECK_EQ(0, sampler->GetSummary().sample_count);
    sampler->Dispose();
  }
  isolate->Dispose();
}

}  // namespace
}  // namespace v8::internal
#endif
