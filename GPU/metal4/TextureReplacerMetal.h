// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/TextureReplacer.h"
#include "GPU/metal4/ReplacedTextureMetal.h"
struct MetalReplacedTextureDecodeInfo {
	u64 cachekey;
	u32 hash;
	u32 addr;
	bool isVideo;
	bool isFinal;
	Draw::DataFormat fmt;
};

class MetalTextureReplacer {
public:
	MetalTextureReplacer(const MetalTextureReplacer &) = delete;
	MetalTextureReplacer &operator=(const MetalTextureReplacer &) = delete;
	// The draw context is checked for supported texture formats.
	MetalTextureReplacer(Draw::DrawContext *draw);
	~MetalTextureReplacer();

	void NotifyConfigChanged();

	bool Enabled() const { return replaceEnabled_ || saveEnabled_; }  // used to check hashing method etc.
	bool ReplaceEnabled() const { return replaceEnabled_; }
	bool SaveEnabled() const { return saveEnabled_; }

	u32 ComputeHash(u32 addr, int bufw, int w, int h, bool swizzled, GETextureFormat fmt, u16 maxSeenV);

	// Returns nullptr if not found.
	MetalReplacedTexture *FindReplacement(ReplacementCacheKey key, int w, int h);

	// For testing: point the replacer at a texture pack directory and load its
	// ini, without touching the global config. Returns true on success.
	bool LoadPackForTesting(const Path &basePath, std::string *error, bool enableSaving = false);

	// Check if a NotifyTextureDecoded for this texture is desired (used to avoid reads from write-combined memory.)
	bool WillSave(const MetalReplacedTextureDecodeInfo &replacedInfo) const;

	// Notify that a new texture was decoded. May already be upscaled, saves the data passed.
	// If the replacer knows about this one already, texture will be passed in, otherwise nullptr.
	void NotifyTextureDecoded(MetalReplacedTexture *texture, const MetalReplacedTextureDecodeInfo &replacedInfo, const void *data, int pitch, int level, int origW, int origH, int scaledW, int scaledH);

	// Whether this texture, which must have a palette (CLUT4 or CLUT8), is saved as an 8-bit palette PNG
	// with NotifyIndexedTextureDecoded instead of an RGBA one, see saveIndexed in textures.ini.
	bool WillSaveIndexed(const MetalReplacedTextureDecodeInfo &replacedInfo) const;
	// Whether the indices of this texture have been saved already, for any of its palettes.
	bool HasSavedIndexed(const MetalReplacedTextureDecodeInfo &replacedInfo) const;
	// Saves the indices (one byte each) with the palette that they get from the game's CLUT, numColors long.
	// The file name has the CLUT hash zeroed and ends in _indexed.
	void NotifyIndexedTextureDecoded(const MetalReplacedTextureDecodeInfo &replacedInfo, const u8 *indices, int pitch, const u32 *palette, int numColors, int w, int h);

	void Decimate(ReplacerDecimateMode mode);

	static bool GenerateIni(const std::string &gameID, Path &generatedFilename);
	static bool IniExists(const std::string &gameID);

	int GetNumTrackedTextures() const { return (int)cache_.size(); }
	int GetNumCachedReplacedTextures() const { return (int)levelCache_.size(); }

	static std::string HashName(ReplacementCacheKey key, int level);

protected:
	bool FindFiltering(ReplacementCacheKey key, TextureFiltering *forceFiltering);
	// How the indices of a replacement from the [indexed] section get their colors.
	enum class IndexedMode : uint8_t {
		NONE,
		CPU,  // Colored with the CLUT when the texture is built.
		GPU,  // Uploaded as indices, colored in the shader.
	};
	IndexedMode LookupIndexed(ReplacementCacheKey key) const;
	// Which textures with a palette are saved as palette PNGs.
	enum class SaveIndexedMode : uint8_t {
		OFF,
		LISTED,  // Only the ones in the [indexed] section.
		ALL,
	};
	ReplacementCacheKey IndexedSaveKey(const MetalReplacedTextureDecodeInfo &replacedInfo) const;

	bool LoadIni(std::string *error, bool notify = true);
	void DeleteVFS();
	bool LoadIniValues(IniFile &ini, VFSBackend *dir, bool isOverride, std::string *error);
	void ParseHashRange(const std::string &key, const std::string &value);
	void ParseFiltering(const std::string &key, const std::string &value);
	void ParseIndexed(const std::string &key, const std::string &value);
	void ParseReduceHashRange(const std::string& key, const std::string& value);
	bool LookupHashRange(u32 addr, int w, int h, int *newW, int *newH);
	float LookupReduceHashRange(int w, int h);
	std::string LookupHashFile(ReplacementCacheKey key, bool *foundAlias, bool *ignored);

	bool replaceEnabled_ = false;
	bool saveEnabled_ = false;
	bool ignoreAddress_ = false;
	bool reduceHash_ = false;
	bool ignoreMipmap_ = false;
	SaveIndexedMode saveIndexed_ = SaveIndexedMode::OFF;
	int skipLastDXT1Blocks128x64_ = 0;
	int skipLastDXT1Blocks128x128_ = 0;

	float reduceHashGlobalValue = 0.5f; // Global value for textures dump pngs of all sizes, 0.5 by default but can be set in textures.ini

	double lastTextureCacheSizeGB_ = 0.0;
	std::string gameID_;
	Path basePath_;
	Path newTextureDir_;
	ReplacedTextureHash textureHash_ = ReplacedTextureHash::QUICK;

	VFSBackend *vfs_ = nullptr;
	bool vfsIsZip_ = false;

	MetalGPUFormatSupport formatSupport_{};

	typedef std::pair<int, int> WidthHeightPair;
	std::unordered_map<u64, WidthHeightPair> hashranges_;
	std::unordered_map<u64, float> reducehashranges_;

	std::unordered_map<ReplacementCacheKey, std::string> aliases_;
	std::unordered_map<ReplacementCacheKey, TextureFiltering> filtering_;
	// Replacements that hold palette indices instead of colors, from the [indexed] section.
	std::unordered_map<ReplacementCacheKey, IndexedMode> indexed_;

	std::unordered_map<ReplacementCacheKey, MetalReplacedTextureRef> cache_;
	std::unordered_map<ReplacementCacheKey, SavedTextureCacheData> savedCache_;

	// the key is either from aliases_, in which case it's a |-separated sequence of texture filenames of the levels of a texture.
	// alternatively the key is from the generated texture filename.
	std::unordered_map<std::string, MetalReplacedTexture *> levelCache_;
};
