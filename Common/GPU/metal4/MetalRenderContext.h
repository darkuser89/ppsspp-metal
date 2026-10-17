// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Common/GPU/metal4/MetalShaderCompiler.h"

namespace Metal {

struct UploadSlice {
	id<MTLBuffer> buffer = nil;
	size_t offset = 0;
	explicit operator bool() const { return buffer != nil; }
};

struct Submission {
	mutable std::mutex mutex;
	std::condition_variable ready;
	bool complete = false;
	std::string error;
	double gpuStart = 0.0;
	double gpuEnd = 0.0;
	uint64_t serial = 0;
	bool IsComplete() const { std::lock_guard<std::mutex> lock(mutex); return complete; }
};

// Encoding belongs to the render thread. Slot resources and allocator storage
// remain alive until Metal 4's commit feedback reports GPU completion.
class RenderContext {
public:
	RenderContext() = default;
	~RenderContext();
	RenderContext(const RenderContext &) = delete;
	RenderContext &operator=(const RenderContext &) = delete;

	bool Init(std::string *error, size_t inflightFrames);
	bool BeginCommands(std::string *error);
	void SetBeginCommandsCallback(std::function<void()> callback) { beginCommandsCallback_ = std::move(callback); }
	void SetProfilingEnabled(bool enabled, bool log) { profilingEnabled_ = enabled; profileLogEnabled_ = log; }
	void ProfileRenderPass(MTL4RenderPassDescriptor *pass, const char *tag);
	id<MTL4RenderCommandEncoder> RenderEncoder(id<MTL4CommandBuffer> commands, MTL4RenderPassDescriptor *pass, const char *tag = nullptr);
	id<MTL4ComputeCommandEncoder> BlitEncoder(id<MTL4CommandBuffer> commands, const char *tag);
	id<MTL4ComputeCommandEncoder> ComputeEncoder(id<MTL4CommandBuffer> commands, const char *tag);
	void EndEncoder(id<MTL4CommandEncoder> encoder);
	const std::string &GPUProfileString() const { return gpuProfileString_; }
	bool SubmitCommands(bool wait, std::string *error);
	bool WaitUntilIdle(std::string *error);
	id<MTL4CommandBuffer> InitializationCommands(std::string *error);
	id<MTL4ComputeCommandEncoder> InitializationBlitEncoder(std::string *error);
	UploadSlice ReserveUpload(size_t size, std::string *error);
	UploadSlice Upload(const void *data, size_t size, std::string *error);
	id<MTLBuffer> AcquireReadbackBuffer(size_t size);
	void RecycleReadbackBuffer(id<MTLBuffer> buffer);

	void Keep(id object);
	void Keep(std::initializer_list<id> objects) { for (id object : objects) { Keep(object); } }
	void SetBuffer(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLBuffer> buffer, size_t offset, uint32_t index);
	void SetBytes(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, const void *bytes, size_t size, uint32_t index);
	void SetTexture(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLTexture> texture, uint32_t index);
	void SetSampler(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLSamplerState> sampler, uint32_t index);
	void SetBuffer(id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> buffer, size_t offset, uint32_t index);
	void SetBytes(id<MTL4ComputeCommandEncoder> encoder, const void *bytes, size_t size, uint32_t index);
	void SetTexture(id<MTL4ComputeCommandEncoder> encoder, id<MTLTexture> texture, uint32_t index);
	void SetSampler(id<MTL4ComputeCommandEncoder> encoder, id<MTLSamplerState> sampler, uint32_t index);
	void PresentDrawable(id<MTLDrawable> drawable);
	std::shared_ptr<Submission> CurrentSubmission() const { return currentCompletion_; }

	id<MTLDevice> Device() const { return device_; }
	id<MTL4Compiler> Compiler() const { return compiler_; }
	MTL4LibraryFunctionDescriptor *FunctionDescriptor(id<MTLFunction> function) const;
	id<MTLComputePipelineState> CreateComputePipeline(id<MTLFunction> function, NSError **error);
	id<MTL4CommandBuffer> Commands() const { return commands_; }
	uint64_t CommandGeneration() const { return commandGeneration_; }
	double LastSubmissionGPUTimeMs() const { return lastSubmissionGPUTimeMs_.load(std::memory_order_relaxed); }
	std::string DeviceName() const;
	id<MTLFunction> CreateShader(const CompiledShader &shader, const char *tag, std::string *error);

private:
	static bool WaitForCommands(const std::shared_ptr<Submission> &completion, std::string *error);
	bool WaitForSubmission(size_t slot, std::string *error);
	void EndInitializationBlit();
	enum class ProfileKind { RENDER, BLIT, COMPUTE };
	NSUInteger ReserveProfileSamples(size_t count, const char *tag, ProfileKind kind);
	id<MTL4ArgumentTable> NewArgumentTable();
	void PrepareEncoder(id<MTL4CommandEncoder> encoder, const char *tag, ProfileKind kind);
	id<MTL4ArgumentTable> Arguments(id<MTL4CommandEncoder> encoder, MTLRenderStages stage);
	struct EncoderState {
		id<MTL4ArgumentTable> vertex = nil;
		id<MTL4ArgumentTable> fragment = nil;
		NSUInteger profileIndex = NSNotFound;
		ProfileKind kind = ProfileKind::COMPUTE;
	};
	std::map<void *, EncoderState> encoderStates_;
	id<MTLDevice> device_ = nil;
	id<MTL4CommandQueue> queue_ = nil;
	id<MTL4Compiler> compiler_ = nil;
	NSMapTable<id<MTLFunction>, id<MTLLibrary>> *functionLibraries_ = nil;
	id<MTL4CommandBuffer> commands_ = nil;
	id<MTL4CommandBuffer> initializationCommands_ = nil;
	id<MTL4ComputeCommandEncoder> initializationBlitEncoder_ = nil;
	std::array<id<MTL4CommandAllocator>, 3> allocators_{};
	std::array<id<MTL4CommandAllocator>, 3> initializationAllocators_{};
	std::array<std::shared_ptr<Submission>, 3> submitted_{};
	std::shared_ptr<Submission> currentCompletion_;
	std::array<id<MTLResidencySet>, 3> residency_{};
	std::array<NSMutableSet *, 3> retained_{};
	std::array<std::vector<id<MTL4ArgumentTable>>, 3> argumentPools_;
	size_t argumentPoolUsed_ = 0;
	id<MTLDrawable> pendingDrawable_ = nil;
	struct ProfilePass {
		std::string tag;
		NSUInteger firstSample = 0;
		ProfileKind kind = ProfileKind::RENDER;
	};
	static constexpr size_t MAX_PROFILE_SAMPLES = 512;
	std::array<id<MTL4CounterHeap>, 3> profileBuffers_{};
	std::array<std::vector<ProfilePass>, 3> submittedProfilePasses_;
	std::array<size_t, 3> submittedProfileSamplesUsed_{};
	std::array<bool, 3> submittedProfileLog_{};
	std::vector<ProfilePass> profilePasses_;
	size_t profileSamplesUsed_ = 0;
	std::string gpuProfileString_;
	bool profilingEnabled_ = false;
	bool profileLogEnabled_ = false;
	std::string nextProfileTag_;
	struct UploadBlock {
		id<MTLBuffer> buffer = nil;
		size_t used = 0;
	};
	std::array<std::vector<UploadBlock>, 3> uploadBlocks_;
	id<MTLBuffer> readbackBuffer_ = nil;
	size_t currentUploadBlock_ = 0;
	size_t nextSubmission_ = 0;
	size_t inflightFrames_ = 2;
	uint64_t commandGeneration_ = 0;
	uint64_t nextSubmissionSerial_ = 0;
	uint64_t lastCompletedSerial_ = 0;
	std::atomic<double> lastSubmissionGPUTimeMs_{-1.0};
	std::function<void()> beginCommandsCallback_;
};

}  // namespace Metal
