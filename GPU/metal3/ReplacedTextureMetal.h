// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/ReplacedTexture.h"
class MetalTextureReplacer;
class MetalReplacedTextureTask;
struct MetalGPUFormatSupport {
	bool metal = false;
	bool bc123;
	bool bc4;
	bool bc5;
	bool astc;
	bool bc7;
	bool etc2;
};

struct MetalReplacementDesc {
	int newW;
	int newH;
	ReplacementCacheKey cacheKey;
	int w;
	int h;
	TextureFiltering forceFiltering;
	std::string hashfiles;
	Path basePath;
	std::vector<std::string> filenames;
	std::string logId;
	MetalGPUFormatSupport formatSupport;
};

class MetalReplacedTexture;

// These aren't actually all replaced, they can also represent a placeholder for a not-found
// replacement (texture == nullptr).
struct MetalReplacedTextureRef {
	MetalReplacedTexture *texture;  // shortcut
	std::string hashfiles;  // key into levelCache_
};

// Metadata about a given texture level.
struct MetalReplacedTextureLevel {
	// Data dimensions
	int w = 0;
	int h = 0;
	// PSP texture dimensions
	int fullW = 0;
	int fullH = 0;

	int fullDataSize = 0;

	// To be able to reload, we need to be able to reopen, unfortunate we can't use zip_file_t.
	// TODO: This really belongs on the level in the cache, not in the individual MetalReplacedTextureLevel objects.
	VFSFileReference *fileRef = nullptr;
};

class MetalReplacedTexture {
public:
	MetalReplacedTexture(const MetalReplacedTexture &) = delete;
	MetalReplacedTexture &operator=(const MetalReplacedTexture &) = delete;
	MetalReplacedTexture(VFSBackend *vfs, const MetalReplacementDesc &desc);
	~MetalReplacedTexture();

	inline ReplacementState State() const {
		return state_;
	}

	void SetState(ReplacementState state) {
		_dbg_assert_(state != state_);
		state_ = state;
	}

	void GetSize(int level, int *w, int *h) const {
		_dbg_assert_(State() == ReplacementState::ACTIVE);
		_dbg_assert_((size_t)level < levels_.size());
		*w = levels_[level].fullW;
		*h = levels_[level].fullH;
	}

	int GetLevelDataSizeAfterCopy(int level) const {
		// Includes padding etc.
		return levels_[level].fullDataSize;
	}

	size_t GetTotalDataSize() const {
		if (State() != ReplacementState::ACTIVE) {
			return 0;
		}
		size_t sz = 0;
		for (auto &data : data_) {
			sz += data.size();
		}
		return sz;
	}

	bool ForceFiltering(TextureFiltering *forceFiltering) const {
		if (desc_.forceFiltering != (TextureFiltering)0) {
			*forceFiltering = desc_.forceFiltering;
			return true;
		} else {
			return false;
		}
	}

	int NumLevels() const {
		_dbg_assert_(State() == ReplacementState::ACTIVE);
		return (int)levels_.size();
	}

	Draw::DataFormat Format() const {
		_dbg_assert_(State() == ReplacementState::ACTIVE);
		return fmt;
	}

	const MetalReplacementDesc &Desc() const {
		return desc_;
	}

	TextureAlpha AlphaStatus() const {
		return alphaStatus_;
	}

	bool Poll(double budget);
	bool CopyLevelTo(int level, uint8_t *out, size_t outDataSize, int rowPitch, bool compressedPitched = false);

	std::string logId_;

private:
	enum class LoadLevelResult {
		LOAD_ERROR = 0,
		CONTINUE = 1,
		DONE = 2,
	};

	void Prepare(VFSBackend *vfs);
	// Waits for any load task, then drops all data and file references, and detaches from the VFS.
	// The texture reloads from scratch once it gets a VFS again. Must run before the VFS is deleted.
	void Unload();
	void ReleaseLevels();
	LoadLevelResult LoadLevelData(VFSFileReference *fileRef, const std::string &filename, int level, Draw::DataFormat *pixelFormat);
	void PurgeIfNotUsedSinceTime(double t);

	std::vector<std::vector<uint8_t>> data_;
	std::vector<MetalReplacedTextureLevel> levels_;
	// Image format of mip level 0; mixing formats across levels is not
	// allowed (container formats like KTX2/DDS manage their own mip chain).
	ReplacedImageType firstImageType_ = ReplacedImageType::INVALID;

	double lastUsed_ = 0.0;
	LimitedWaitable *threadWaitable_ = nullptr;
	std::mutex lock_;
	Draw::DataFormat fmt = Draw::DataFormat::UNDEFINED;
	TextureAlpha alphaStatus_ = TextureAlpha::Any;
	double lastUsed = 0.0;

	std::atomic<ReplacementState> state_ = ReplacementState::UNLOADED;

	VFSBackend *vfs_ = nullptr;
	MetalReplacementDesc desc_;

	friend class MetalTextureReplacer;
	friend class MetalReplacedTextureTask;
};
