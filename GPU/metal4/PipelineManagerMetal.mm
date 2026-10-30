// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/metal4/PipelineManagerMetal.h"

#include <algorithm>
#include <iterator>
#include "Common/File/FileUtil.h"
#include "Common/Log.h"
#include "Common/TimeUtil.h"
#include "Core/Config.h"
#include "GPU/Common/ShaderCommon.h"
#include "GPU/GPUDefinitions.h"
#include "GPU/GPUState.h"
#include "ext/xxhash.h"

#include <climits>
#include <vector>

namespace {

constexpr uint32_t METAL_PIPELINE_CACHE_MAGIC = 0x4D50534F;
constexpr uint32_t METAL_PIPELINE_CACHE_VERSION = PPSSPP_PLATFORM(MAC) ? 7 : 5;
constexpr uint32_t MAX_BINARY_ARCHIVES = 8;
constexpr uint32_t MAX_CACHED_PIPELINES = 8192;

struct MetalPipelineCacheHeader {
	uint32_t magic;
	uint32_t version;
	uint32_t useFlags;
	uint32_t vendorChecks;
	uint64_t deviceHash;
	uint64_t buildHash;
	uint32_t count;
#if PPSSPP_PLATFORM(MAC)
	uint32_t archiveBank;
	uint32_t archiveCount;
	std::array<uint64_t, MAX_BINARY_ARCHIVES> archiveHashes;
#endif
};

static_assert(sizeof(MetalPipelineCacheHeader) == (PPSSPP_PLATFORM(MAC) ? 112 : 40));

Path BinaryArchivePath(const Path &filename, uint32_t bank, uint32_t index) {
	return filename.WithExtraExtension(".metal4archive." + std::to_string(bank) + "." + std::to_string(index));
}

#if PPSSPP_PLATFORM(MAC)
bool BinaryArchiveHash(const Path &filename, uint64_t *hash) {
	FILE *file = File::OpenCFile(filename, "rb");
	if (!file) {
		return false;
	}
	XXH3_state_t *state = XXH3_createState();
	if (!state) {
		fclose(file);
		return false;
	}
	XXH3_64bits_reset(state);
	uint8_t buffer[65536];
	size_t size;
	uint64_t bytes = 0;
	while ((size = fread(buffer, 1, sizeof(buffer), file)) != 0) {
		XXH3_64bits_update(state, buffer, size);
		bytes += size;
	}
	const bool valid = bytes != 0 && !ferror(file);
	*hash = XXH3_64bits_digest(state);
	XXH3_freeState(state);
	fclose(file);
	return valid;
}
#endif

uint64_t CacheStringHash(const char *value) {
	uint64_t hash = 14695981039346656037ULL;
	for (; *value; ++value) {
		hash = (hash ^ (uint8_t)*value) * 1099511628211ULL;
	}
	return hash;
}

struct VertexComponent {
	MTLVertexFormat format;
	uint32_t size;
};

// Decoder components are padded to 4 bytes (8 for short3). Match the existing
// Vulkan fetch conventions, including normalized packed colors and normals.
constexpr VertexComponent COMPONENTS[] = {
	{MTLVertexFormatInvalid, 0},
	{MTLVertexFormatFloat, 4},
	{MTLVertexFormatFloat2, 8},
	{MTLVertexFormatFloat3, 12},
	{MTLVertexFormatFloat4, 16},
	{MTLVertexFormatChar4Normalized, 4},
	{MTLVertexFormatShort4Normalized, 8},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUChar4Normalized, 4},
	{MTLVertexFormatUShort2Normalized, 4},
	{MTLVertexFormatUShort2Normalized, 4},
	{MTLVertexFormatUShort4Normalized, 8},
	{MTLVertexFormatUShort4Normalized, 8},
};
static_assert(ARRAY_SIZE(COMPONENTS) == DEC_U16_4 + 1);

bool SetAttribute(MTLVertexDescriptor *layout, PspAttributeLocation location, uint8_t component, uint32_t offset, uint32_t stride) {
	if (!component) {
		return true;
	}
	if (component >= ARRAY_SIZE(COMPONENTS) || offset > stride || COMPONENTS[component].size > stride - offset) {
		return false;
	}
	auto attr = layout.attributes[(int)location];
	attr.format = COMPONENTS[component].format;
	attr.offset = offset;
	attr.bufferIndex = Metal::VERTEX_BUFFER_SLOT;
	return true;
}

MTLVertexDescriptor *MakeLayout(const MetalGEVertexShader &shader, const DecVtxFormat *decoded, uint32_t *stride, std::string *error) {
	MTLVertexDescriptor *layout = [MTLVertexDescriptor vertexDescriptor];
	bool valid;
	if (shader.UseHWTransform()) {
		if (!decoded || !decoded->stride) {
			*error = "Metal hardware transform requires a decoded vertex layout";
			return nil;
		}
		*stride = decoded->stride;
		valid = SetAttribute(layout, PspAttributeLocation::POSITION, DEC_FLOAT_3, decoded->posoff, *stride) &&
			SetAttribute(layout, PspAttributeLocation::TEXCOORD, decoded->uvfmt, decoded->uvoff, *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR0, decoded->c0fmt, decoded->c0off, *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR1, decoded->c1fmt, decoded->c1off, *stride) &&
			SetAttribute(layout, PspAttributeLocation::NORMAL, decoded->nrmfmt, decoded->nrmoff, *stride) &&
			SetAttribute(layout, PspAttributeLocation::W1, decoded->w0fmt, decoded->w0off, *stride) &&
			SetAttribute(layout, PspAttributeLocation::W2, decoded->w1fmt, decoded->w1off, *stride);
	} else {
		*stride = sizeof(TransformedVertex);
		valid = SetAttribute(layout, PspAttributeLocation::POSITION, DEC_FLOAT_4, offsetof(TransformedVertex, pos), *stride) &&
			SetAttribute(layout, PspAttributeLocation::TEXCOORD, DEC_FLOAT_3, offsetof(TransformedVertex, uv), *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR0, DEC_U8_4, offsetof(TransformedVertex, color0_32), *stride) &&
			SetAttribute(layout, PspAttributeLocation::COLOR1, DEC_U8_4, offsetof(TransformedVertex, color1_32), *stride) &&
			SetAttribute(layout, PspAttributeLocation::NORMAL, DEC_FLOAT_1, offsetof(TransformedVertex, fog), *stride);
	}
	if (!valid) {
		*error = "Metal decoded vertex component is invalid or extends beyond the stride";
		return nil;
	}
	for (MTLAttribute *attr in shader.function.stageInputAttributes) {
		if (attr.active && layout.attributes[attr.attributeIndex].format == MTLVertexFormatInvalid) {
			*error = "Metal vertex shader requires an attribute missing from the decoded layout";
			return nil;
		}
	}
	layout.layouts[Metal::VERTEX_BUFFER_SLOT].stride = *stride;
	layout.layouts[Metal::VERTEX_BUFFER_SLOT].stepFunction = MTLVertexStepFunctionPerVertex;
	return layout;
}

}

MetalPipelineMetrics PipelineManagerMetal::Metrics() const {
	std::lock_guard<std::mutex> lock(metrics_->mutex);
	return metrics_->value;
}

bool PipelineManagerMetal::PipelinesPending() const {
	for (const auto &entry : pending_) {
		std::lock_guard<std::mutex> lock(entry.second->mutex);
		if (!entry.second->complete) {
			return true;
		}
	}
	return false;
}

void PipelineManagerMetal::Clear() {
	lastPipeline_ = nullptr;
	pipelines_.clear();
	pending_.clear();
	depthStates_.clear();
	shaderOwner_ = nullptr;
	shaderGeneration_ = 0;
#if PPSSPP_PLATFORM(MAC)
	binaryArchives_ = nil;
	binaryArchiveKeys_.clear();
#endif
}

void PipelineManagerMetal::DeviceLost() {
	Clear();
	manager_ = nullptr;
}

void PipelineManagerMetal::DeviceRestore(Metal::RenderManager *manager) {
	Clear();
	manager_ = manager;
}

bool PipelineManagerMetal::LoadCache(const Path &filename, ShaderManagerMetal &shaders) {
	if (!g_Config.bShaderCache || !manager_ || !filename.Valid()) {
		return false;
	}
#if PPSSPP_PLATFORM(MAC)
	binaryArchives_ = nil;
	binaryArchiveKeys_.clear();
#endif
	// Reject damaged IDs before invoking shader generation or native compilation.
	const uint64_t bytes = File::GetFileSize(filename);
	if (bytes < sizeof(MetalPipelineCacheHeader) + 8 || bytes > sizeof(MetalPipelineCacheHeader) + MAX_CACHED_PIPELINES * sizeof(PipelineKey) + 8 ||
		!VerifyShaderCacheChecksum(filename)) {
		return false;
	}
	FILE *file = File::OpenCFile(filename, "rb");
	if (!file) {
		return false;
	}
	MetalPipelineCacheHeader header{};
	bool valid = fread(&header, sizeof(header), 1, file) == 1 &&
		header.magic == METAL_PIPELINE_CACHE_MAGIC && header.version == METAL_PIPELINE_CACHE_VERSION &&
		header.useFlags == gstate_c.GetUseFlags() && header.vendorChecks == (uint32_t)g_Config.bVendorBugChecksEnabled &&
		header.deviceHash == CacheStringHash(manager_->Context().DeviceName().c_str()) &&
		header.buildHash == CacheStringHash(PPSSPP_GIT_VERSION) && header.count <= MAX_CACHED_PIPELINES &&
		bytes == sizeof(header) + header.count * sizeof(PipelineKey) + 8;
#if PPSSPP_PLATFORM(MAC)
	valid &= header.archiveBank <= 1 && header.archiveCount <= MAX_BINARY_ARCHIVES;
#endif
	std::vector<PipelineKey> keys;
	if (valid) {
		keys.resize(header.count);
		valid = keys.empty() || fread(keys.data(), sizeof(PipelineKey), keys.size(), file) == keys.size();
	}
	fclose(file);
	if (!valid) {
		return false;
	}
#if PPSSPP_PLATFORM(MAC)
	NSMutableArray<id<MTL4Archive>> *archives = [NSMutableArray new];
	for (uint32_t i = 0; i < header.archiveCount; ++i) {
		const Path path = BinaryArchivePath(filename, header.archiveBank, i);
		uint64_t hash;
		// Validate opaque archive bytes before passing them to the native loader.
		if (!BinaryArchiveHash(path, &hash) || hash != header.archiveHashes[i]) {
			WARN_LOG(Log::G3D, "Ignoring missing or damaged Metal GE binary archives");
			[archives removeAllObjects];
			break;
		}
		NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
		NSError *nativeError = nil;
		id<MTL4Archive> archive = [manager_->Context().Device() newArchiveWithURL:url error:&nativeError];
		if (!archive) {
			WARN_LOG(Log::G3D, "Ignoring incomplete Metal GE binary archives: %s", nativeError.localizedDescription.UTF8String ?: "missing file");
			[archives removeAllObjects];
			break;
		}
		[archives addObject:archive];
	}
	if (archives.count) {
		binaryArchives_ = archives;
		binaryArchiveBank_ = header.archiveBank;
		binaryArchiveCachePath_ = filename;
		binaryArchiveKeys_ = keys;
		binaryArchiveHashes_ = header.archiveHashes;
		std::sort(binaryArchiveKeys_.begin(), binaryArchiveKeys_.end());
	}
#endif
	const double start = time_now_d();
	uint32_t scheduled = 0;
	bool failed = false;
	for (const PipelineKey &key : keys) {
		// Cache files can be truncated or corrupted without invalidating the header.
		// Metal asserts on invalid blend enums before returning a pipeline error.
		bool invalid = key[3] != 0 || key[4] > 1 || key[7] > MTLBlendOperationMax || key[10] > MTLBlendOperationMax ||
			key[11] > MTLColorWriteMaskAll || key[12] > 0xFF || key[13] > 0xFF ||
			key[20] == 0 || key[20] > INT_MAX;
		for (int i : {5, 6, 8, 9}) {
			invalid |= key[i] > MTLBlendFactorOneMinusSource1Alpha;
		}
		for (int i = 14; i <= 19; ++i) {
			invalid |= key[i] > 0xFFFF;
		}
		if (invalid) {
			WARN_LOG(Log::G3D, "Ignoring invalid Metal GE pipeline cache entry");
			failed = true;
			continue;
		}
		VShaderID vertexID;
		FShaderID fragmentID;
		vertexID.FromUint64(key[0]);
		fragmentID.FromUint64(key[1]);
		MetalGEBlendState blend;
		blend.enabled = key[4] != 0;
		blend.srcColor = (MTLBlendFactor)key[5];
		blend.dstColor = (MTLBlendFactor)key[6];
		blend.colorOp = (MTLBlendOperation)key[7];
		blend.srcAlpha = (MTLBlendFactor)key[8];
		blend.dstAlpha = (MTLBlendFactor)key[9];
		blend.alphaOp = (MTLBlendOperation)key[10];
		blend.writeMask = (MTLColorWriteMask)key[11];
		DecVtxFormat decoded{};
		decoded.stride = (uint8_t)key[12];
		decoded.posoff = (uint8_t)key[13];
		decoded.uvfmt = (uint8_t)key[14];
		decoded.uvoff = (uint8_t)(key[14] >> 8);
		decoded.c0fmt = (uint8_t)key[15];
		decoded.c0off = (uint8_t)(key[15] >> 8);
		decoded.c1fmt = (uint8_t)key[16];
		decoded.c1off = (uint8_t)(key[16] >> 8);
		decoded.nrmfmt = (uint8_t)key[17];
		decoded.nrmoff = (uint8_t)(key[17] >> 8);
		decoded.w0fmt = (uint8_t)key[18];
		decoded.w0off = (uint8_t)(key[18] >> 8);
		decoded.w1fmt = (uint8_t)key[19];
		decoded.w1off = (uint8_t)(key[19] >> 8);
		std::string error;
		GetOrCreateInternal(shaders, vertexID, fragmentID, vertexID.Bit(VS_BIT_USE_HW_TRANSFORM) ? &decoded : nullptr,
			blend, (MTLPixelFormat)key[2], (MTLPixelFormat)key[3], &error, (int)key[20], true);
		scheduled += error.empty();
		failed |= !error.empty();
	}
	if (failed) {
		// A bad prewarm entry must not cache a failed shader for later draws.
		shaders.DiscardFailedShaders();
	}
	if (scheduled && !failed) {
		gstate_c.useFlagsChanged = false;
	}
	INFO_LOG(Log::G3D, "Scheduled %u/%u Metal GE pipelines in %.1f ms", scheduled, header.count, (time_now_d() - start) * 1000.0);
	return !failed;
}

void PipelineManagerMetal::SaveCache(const Path &filename) const {
	if (!g_Config.bShaderCache || !manager_ || !filename.Valid() || gstate_c.useFlagsChanged ||
		(shaderOwner_ && (shaderGeneration_ != shaderOwner_->CacheGeneration() || !shaderOwner_->CacheMatchesEnvironment()))) {
		return;
	}
	struct SaveTimer {
		std::shared_ptr<MetricsState> metrics;
		double start = time_now_d();
		~SaveTimer() {
			std::lock_guard<std::mutex> lock(metrics->mutex);
			metrics->value.cacheSaveSeconds += time_now_d() - start;
			++metrics->value.cacheSaves;
		}
	} timer{metrics_};
	{
		std::lock_guard<std::mutex> lock(metrics_->mutex);
		// Discarded generations can still have native compiler callbacks in flight.
		// Defer file replacement until all users of the loaded archives finish.
		if (metrics_->activeCompiles) {
			++metrics_->value.archiveDeferrals;
			return;
		}
	}
	std::vector<PipelineKey> keys;
	keys.reserve(pipelines_.size() + pending_.size());
	for (const auto &entry : pipelines_) {
		keys.push_back(entry.first);
	}
	for (const auto &entry : pending_) {
		std::lock_guard<std::mutex> lock(entry.second->mutex);
		if (entry.second->complete && entry.second->pipeline.state) {
			keys.push_back(entry.first);
		}
	}
	std::sort(keys.begin(), keys.end());
#if PPSSPP_PLATFORM(MAC)
	bool haveArchives = binaryArchiveCachePath_ == filename && binaryArchives_.count > 0;
	for (NSUInteger i = 0; haveArchives && i < binaryArchives_.count; ++i) {
		haveArchives = File::Exists(BinaryArchivePath(filename, binaryArchiveBank_, i));
	}
	if (haveArchives) {
		// Preserve already archived IDs even if a failed prewarm has not been drawn.
		keys.insert(keys.end(), binaryArchiveKeys_.begin(), binaryArchiveKeys_.end());
		std::sort(keys.begin(), keys.end());
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	}
#endif
	if (keys.size() > MAX_CACHED_PIPELINES) {
		return;
	}
	MetalPipelineCacheHeader header{};
	header.magic = METAL_PIPELINE_CACHE_MAGIC;
	header.version = METAL_PIPELINE_CACHE_VERSION;
	header.useFlags = gstate_c.GetUseFlags();
	header.vendorChecks = g_Config.bVendorBugChecksEnabled;
	header.deviceHash = CacheStringHash(manager_->Context().DeviceName().c_str());
	header.buildHash = CacheStringHash(PPSSPP_GIT_VERSION);
	header.count = (uint32_t)keys.size();
#if PPSSPP_PLATFORM(MAC)
	std::vector<PipelineKey> additions;
	if (haveArchives) {
		std::set_difference(keys.begin(), keys.end(), binaryArchiveKeys_.begin(), binaryArchiveKeys_.end(), std::back_inserter(additions));
	} else {
		additions = keys;
	}
	const bool compact = !haveArchives || (!additions.empty() && binaryArchives_.count >= MAX_BINARY_ARCHIVES);
	uint32_t bank = binaryArchiveBank_;
	bool preservePreviousArchive = haveArchives;
	NSMutableArray<id<MTL4Archive>> *archives = haveArchives ? [binaryArchives_ mutableCopy] : [NSMutableArray new];
	header.archiveHashes = haveArchives ? binaryArchiveHashes_ : std::array<uint64_t, MAX_BINARY_ARCHIVES>{};
	if (compact) {
		// Write into the other bank, preserving the last complete cache on failure.
		MetalPipelineCacheHeader previous{};
		FILE *old = File::OpenCFile(filename, "rb");
		if (old) {
			if (fread(&previous, sizeof(previous), 1, old) == 1 && previous.magic == METAL_PIPELINE_CACHE_MAGIC &&
				previous.version == METAL_PIPELINE_CACHE_VERSION && previous.archiveBank <= 1) {
				bank = previous.archiveBank;
				preservePreviousArchive |= previous.archiveCount > 0 && previous.archiveCount <= MAX_BINARY_ARCHIVES && VerifyShaderCacheChecksum(filename);
			}
			fclose(old);
		}
		bank = 1 - bank;
		archives = [NSMutableArray new];
		header.archiveHashes = {};
		additions = keys;
	}
	Path createdArchive;
	if (!additions.empty()) {
		const double start = time_now_d();
		NSError *nativeError = nil;
		MTL4PipelineDataSetSerializerDescriptor *serializerDesc = [MTL4PipelineDataSetSerializerDescriptor new];
		serializerDesc.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
		id<MTL4PipelineDataSetSerializer> serializer = [manager_->Context().Device() newPipelineDataSetSerializerWithDescriptor:serializerDesc];
		MTL4CompilerDescriptor *compilerDesc = [MTL4CompilerDescriptor new];
		compilerDesc.pipelineDataSetSerializer = serializer;
		id<MTL4Compiler> compiler = [manager_->Context().Device() newCompilerWithDescriptor:compilerDesc error:&nativeError];
		MTL4CompilerTaskOptions *options = [MTL4CompilerTaskOptions new];
		options.lookupArchives = haveArchives ? binaryArchives_ : nil;
		bool captured = serializer && compiler;
		uint64_t capturedPipelines = 0;
		for (const auto &key : additions) {
			if (!captured) {
				break;
			}
			MTL4RenderPipelineDescriptor *descriptor = nil;
			auto ready = pipelines_.find(key);
			if (ready != pipelines_.end()) {
				descriptor = ready->second.descriptor;
			} else {
				auto pending = pending_.find(key);
				if (pending != pending_.end()) {
					std::lock_guard<std::mutex> lock(pending->second->mutex);
					descriptor = pending->second->pipeline.descriptor;
				}
			}
			if (descriptor) {
				++capturedPipelines;
				captured = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:options error:&nativeError] != nil;
			} else {
				captured = false;
			}
		}
		const Path archivePath = BinaryArchivePath(filename, bank, (uint32_t)archives.count);
		const Path temporary = archivePath.WithExtraExtension(".tmp");
		NSURL *tempURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:temporary.c_str()]];
		bool archived = captured && !(File::Exists(temporary) && File::IsDirectory(temporary)) && [serializer serializeAsArchiveAndFlushToURL:tempURL error:&nativeError] &&
			BinaryArchiveHash(temporary, &header.archiveHashes[archives.count]) && File::Rename(temporary, archivePath);
		id<MTL4Archive> archive = nil;
		if (archived) {
			NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archivePath.c_str()]];
			archive = [manager_->Context().Device() newArchiveWithURL:url error:&nativeError];
			archived = archive != nil;
			if (!archived) {
				File::Delete(archivePath, true);
			}
		}
		{
			std::lock_guard<std::mutex> lock(metrics_->mutex);
			metrics_->value.archiveSeconds += time_now_d() - start;
			++metrics_->value.archiveSaves;
			metrics_->value.archivePipelines += capturedPipelines;
			metrics_->value.archiveFailures += !archived;
		}
		if (!archived) {
			File::Delete(temporary, true);
			WARN_LOG(Log::G3D, "Could not save Metal 4 GE binary archive: %s", nativeError.localizedDescription.UTF8String ?: "archive capture or file I/O failed");
			if (preservePreviousArchive) {
				return;
			}
			// Native archives are optional: retain IDs for compiler prewarming when
			// the driver cannot create a usable archive and no complete bank exists.
			archives = [NSMutableArray new];
			header.archiveHashes = {};
		} else {
			createdArchive = archivePath;
			[archives addObject:archive];
		}
	}
	header.archiveBank = bank;
	header.archiveCount = (uint32_t)archives.count;
#endif
	const Path temporary = filename.WithExtraExtension(".tmp");
	FILE *file = File::OpenCFile(temporary, "wb");
	bool valid = false;
	if (file) {
		valid = fwrite(&header, sizeof(header), 1, file) == 1;
		for (const auto &key : keys) {
			valid &= fwrite(key.data(), sizeof(uint64_t), key.size(), file) == key.size();
		}
		valid &= fclose(file) == 0;
		valid = valid && AppendShaderCacheChecksum(temporary) && File::Rename(temporary, filename);
	}
	if (!valid) {
		File::Delete(temporary, true);
#if PPSSPP_PLATFORM(MAC)
		if (createdArchive.Valid()) {
			File::Delete(createdArchive, true);
		}
#endif
		return;
	}
#if PPSSPP_PLATFORM(MAC)
	binaryArchives_ = archives;
	binaryArchiveBank_ = bank;
	binaryArchiveCachePath_ = filename;
	binaryArchiveKeys_ = std::move(keys);
	binaryArchiveHashes_ = header.archiveHashes;
	// Metadata now refers only to the complete new bank and its published parts.
	for (uint32_t oldBank = 0; oldBank < 2; ++oldBank) {
		for (uint32_t i = 0; i < MAX_BINARY_ARCHIVES; ++i) {
			if (oldBank != bank || i >= archives.count) {
				File::Delete(BinaryArchivePath(filename, oldBank, i), true);
			}
		}
	}
	File::Delete(filename.WithExtraExtension(".metal4archive"), true);
#endif
}

const MetalGEPipeline *PipelineManagerMetal::GetOrCreate(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
	const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
	MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount) {
	return GetOrCreateInternal(shaders, vertexID, fragmentID, decoded, blend, colorFormat, depthStencilFormat, error, sampleCount, false);
}

const MetalGEPipeline *PipelineManagerMetal::Request(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
	const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
	MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount) {
	return GetOrCreateInternal(shaders, vertexID, fragmentID, decoded, blend, colorFormat, depthStencilFormat, error, sampleCount, true);
}

const MetalGEPipeline *PipelineManagerMetal::GetOrCreateInternal(ShaderManagerMetal &shaders, VShaderID vertexID, FShaderID fragmentID,
	const DecVtxFormat *decoded, const MetalGEBlendState &blend, MTLPixelFormat colorFormat,
	MTLPixelFormat depthStencilFormat, std::string *error, int sampleCount, bool async) {
	error->clear();
	if (!manager_) {
		*error = "Metal pipeline manager has no rendering device";
		return nullptr;
	}
	if (sampleCount <= 0 || ![manager_->Context().Device() supportsTextureSampleCount:sampleCount]) {
		*error = "Unsupported Metal GE sample count";
		return nullptr;
	}
	// These are the attachment formats currently allocated by Metal Framebuffer.
	if ((colorFormat != MTLPixelFormatRGBA8Unorm && colorFormat != MTLPixelFormatBGRA8Unorm) ||
		(depthStencilFormat != MTLPixelFormatInvalid && depthStencilFormat != MTLPixelFormatDepth32Float_Stencil8)) {
		*error = "Unsupported Metal GE render target format";
		return nullptr;
	}
	const MetalGEVertexShader *vertex = nullptr;
	const MetalGEFragmentShader *fragment = nullptr;
	if (!shaders.CacheMatchesEnvironment()) {
		// Let the shader manager update its generation before checking pipeline keys.
		vertex = shaders.GetVertexShaderFromID(vertexID, error);
		if (!vertex) {
			return nullptr;
		}
		fragment = shaders.GetFragmentShaderFromID(fragmentID, error);
		if (!fragment) {
			return nullptr;
		}
	}
	if (shaderOwner_ != &shaders || shaderGeneration_ != shaders.CacheGeneration()) {
#if PPSSPP_PLATFORM(MAC)
		if (shaderOwner_) {
			binaryArchives_ = nil;
			binaryArchiveKeys_.clear();
		}
#endif
		lastPipeline_ = nullptr;
		pipelines_.clear();
		pending_.clear();
		shaderOwner_ = &shaders;
		shaderGeneration_ = shaders.CacheGeneration();
	}
	const bool hardware = vertexID.Bit(VS_BIT_USE_HW_TRANSFORM);
	if (hardware && !decoded) {
		*error = "Metal hardware transform requires a decoded vertex layout";
		return nullptr;
	}
	// Metal 4 PSOs have no depth/stencil attachment format. Keep that cache field
	// reserved; attachment-dependent tests and writes are dynamic encoder state.
	PipelineKey key{vertexID.ToUint64(), fragmentID.ToUint64(), (uint64_t)colorFormat, 0,
		blend.enabled, blend.srcColor, blend.dstColor, blend.colorOp, blend.srcAlpha, blend.dstAlpha, blend.alphaOp, blend.writeMask};
	key[20] = sampleCount;
	if (hardware) {
		// The decoder ID normally implies offsets. Key the actual layout too, so
		// independently constructed layouts cannot silently reuse the wrong PSO.
		key[12] = decoded->stride;
		key[13] = decoded->posoff;
		key[14] = decoded->uvfmt | (decoded->uvoff << 8);
		key[15] = decoded->c0fmt | (decoded->c0off << 8);
		key[16] = decoded->c1fmt | (decoded->c1off << 8);
		key[17] = decoded->nrmfmt | (decoded->nrmoff << 8);
		key[18] = decoded->w0fmt | (decoded->w0off << 8);
		key[19] = decoded->w1fmt | (decoded->w1off << 8);
	}
	// All environment, generation, target and layout checks above still apply.
	if (lastPipeline_ && lastKey_ == key) {
		return lastPipeline_;
	}
	auto remember = [&](const MetalGEPipeline *pipeline) {
		lastKey_ = key;
		lastPipeline_ = pipeline;
		return pipeline;
	};
	auto found = pipelines_.find(key);
	if (found != pipelines_.end()) {
		return remember(&found->second);
	}
	auto pending = pending_.find(key);
	if (pending != pending_.end()) {
		if (async) {
			return nullptr;
		}
		auto request = pending->second;
		std::unique_lock<std::mutex> lock(request->mutex);
		if (!request->complete) {
			const double start = time_now_d();
			request->ready.wait(lock, [&] { return request->complete; });
			std::lock_guard<std::mutex> metricsLock(metrics_->mutex);
			metrics_->value.waitSeconds += time_now_d() - start;
			++metrics_->value.waits;
		}
		MetalGEPipeline result = request->pipeline;
		*error = request->error;
		lock.unlock();
		pending_.erase(pending);
		if (result.state) {
			return remember(&pipelines_.emplace(key, std::move(result)).first->second);
		}
		// A failed asynchronous prewarm can be transient. Retry once on the draw
		// thread, so the first draw does not silently disappear.
		error->clear();
	}
	if (!vertex) {
		vertex = shaders.GetVertexShaderFromID(vertexID, error);
		if (!vertex) {
			return nullptr;
		}
		fragment = shaders.GetFragmentShaderFromID(fragmentID, error);
		if (!fragment) {
			return nullptr;
		}
	}
	MetalGEPipeline result;
	MTLVertexDescriptor *layout = MakeLayout(*vertex, decoded, &result.stride, error);
	if (!layout) {
		return nullptr;
	}
	MTL4RenderPipelineDescriptor *desc = [MTL4RenderPipelineDescriptor new];
	desc.label = @"PSP GE pipeline";
	desc.vertexFunctionDescriptor = manager_->Context().FunctionDescriptor(vertex->function);
	desc.fragmentFunctionDescriptor = manager_->Context().FunctionDescriptor(fragment->function);
	desc.vertexDescriptor = layout;
	// GE submits triangles (or triangle strips) after primitive conversion.
	// Metal requires this when a multiview vertex shader writes a target layer.
	desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;


	desc.rasterSampleCount = sampleCount;
	auto color = desc.colorAttachments[0];
	color.pixelFormat = colorFormat;
	color.blendingState = blend.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
	color.sourceRGBBlendFactor = blend.srcColor;
	color.destinationRGBBlendFactor = blend.dstColor;
	color.rgbBlendOperation = blend.colorOp;
	color.sourceAlphaBlendFactor = blend.srcAlpha;
	color.destinationAlphaBlendFactor = blend.dstAlpha;
	color.alphaBlendOperation = blend.alphaOp;
	color.writeMask = blend.writeMask;
	for (const auto &resource : fragment->compiled.resources) {
		if (resource.kind == Metal::ResourceKind::SampledTexture) {
			result.textureMask |= 1u << resource.index;
		}
	}
#if PPSSPP_PLATFORM(MAC)
	if (g_Config.bShaderCache) {
		result.descriptor = [desc copy];
	}

#endif
	MTL4CompilerTaskOptions *taskOptions = [[MTL4CompilerTaskOptions alloc] init];
	taskOptions.lookupArchives = binaryArchives_;
	if (async) {
		auto request = std::make_shared<PendingPipeline>();
		request->pipeline = result;
		pending_.emplace(key, request);
		const auto metrics = metrics_;
		{
			std::lock_guard<std::mutex> lock(metrics->mutex);
			++metrics->activeCompiles;
		}
		const double start = time_now_d();
		[manager_->Context().Compiler() newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:taskOptions completionHandler:^(id<MTLRenderPipelineState> state, NSError *nativeError) {
			@autoreleasepool {
				std::lock_guard<std::mutex> lock(request->mutex);
				request->pipeline.state = state;
				if (!state) {
					const char *description = nativeError.localizedDescription.UTF8String;
					request->error = description ? description : "Failed to create Metal GE pipeline";
				}
				{
					std::lock_guard<std::mutex> metricsLock(metrics->mutex);
					metrics->value.asyncSeconds += time_now_d() - start;
					++metrics->value.asyncCompiles;
					--metrics->activeCompiles;
				}
				request->complete = true;
			}
			request->ready.notify_all();
		}];
		return nullptr;
	}
	NSError *nativeError = nil;
	const double start = time_now_d();
	result.state = [manager_->Context().Compiler() newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:taskOptions error:&nativeError];
	{
		std::lock_guard<std::mutex> lock(metrics_->mutex);
		metrics_->value.syncSeconds += time_now_d() - start;
		++metrics_->value.syncCompiles;
	}
	if (!result.state) {
		*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to create Metal GE pipeline";
		return nullptr;
	}
	return remember(&pipelines_.emplace(key, std::move(result)).first->second);
}

id<MTLDepthStencilState> PipelineManagerMetal::GetDepthStencil(const MetalGEDepthStencilState &state, std::string *error) {
	error->clear();
	if (!manager_) {
		*error = "Metal pipeline manager has no rendering device";
		return nil;
	}
	DepthKey key{state.depthWrite, state.depthCompare, state.stencilEnabled, state.stencilCompare,
		state.stencilFail, state.depthFail, state.pass, state.readMask, state.writeMask};
	auto found = depthStates_.find(key);
	if (found != depthStates_.end()) {
		return found->second;
	}
	MTLDepthStencilDescriptor *desc = [MTLDepthStencilDescriptor new];
	desc.label = @"PSP GE depth/stencil";
	desc.depthWriteEnabled = state.depthWrite;
	desc.depthCompareFunction = state.depthCompare;
	if (state.stencilEnabled) {
		MTLStencilDescriptor *stencil = [MTLStencilDescriptor new];
		stencil.stencilCompareFunction = state.stencilCompare;
		stencil.stencilFailureOperation = state.stencilFail;
		stencil.depthFailureOperation = state.depthFail;
		stencil.depthStencilPassOperation = state.pass;
		stencil.readMask = state.readMask;
		stencil.writeMask = state.writeMask;
		desc.frontFaceStencil = stencil;
		desc.backFaceStencil = stencil;
	}
	auto native = [manager_->Context().Device() newDepthStencilStateWithDescriptor:desc];
	if (!native) {
		*error = "Failed to create Metal GE depth/stencil state";
		return nil;
	}
	depthStates_.emplace(key, native);
	return native;
}
