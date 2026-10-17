// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

#pragma once
#include "GPU/Common/PresentationCommon.h"
#include "GPU/metal4/PostShaderMetal.h"
class PresentationMetal : public PresentationCommon {
public:
	PresentationMetal(const PresentationMetal &) = delete;
	PresentationMetal &operator=(const PresentationMetal &) = delete;
	PresentationMetal(Draw::DrawContext *draw);
	~PresentationMetal();

	void UpdateDisplaySize(int w, int h) {
		pixelWidth_ = w;
		pixelHeight_ = h;
	}

	// NOTE: Should be un-rotated width/height.
	void UpdateRenderSize(int rw, int rh) {
		renderWidth_ = rw;
		renderHeight_ = rh;
	}
	void SetLanguage(ShaderLanguage lang) {
		lang_ = lang;
	}

	bool HasPostShader() {
		return usePostShader_;
	}

	bool UpdatePostShader(const DisplayLayoutConfig &config);

	void BeginFrame(const DisplayLayoutConfig &config) {
		if (restorePostShader_) {
			UpdatePostShader(config);
			restorePostShader_ = false;
		}
		presentedThisFrame_ = false;
	}
	bool PresentedThisFrame() const {
		return presentedThisFrame_;
	}
	void NotifyPresent() {
		// Something else did the present, skipping PresentationCommon.
		// If you haven't called BindFramebufferAsRenderTarget, you must not set this.
		presentedThisFrame_ = true;
	}

	void DeviceLost();
	void DeviceRestore(Draw::DrawContext *draw);

	void UpdateUniforms(bool hasVideo);

	// One of these must be called every frame.
	void SourceBlank();
	void SourceTexture(Draw::Texture *texture, int bufferWidth, int bufferHeight);
	void SourceFramebuffer(Draw::Framebuffer *fb, int bufferWidth, int bufferHeight);

	void RunPostshaderPasses(const DisplayLayoutConfig &config, OutputFlags flags, int uvRotation, float u0, float v0, float u1, float v1);
	void CopyToOutput(const DisplayLayoutConfig &config);

	void CalculateRenderResolution(const DisplayLayoutConfig &config, int *width, int *height, int *scaleFactor, bool *upscaling, bool *ssaa) const;

protected:
	struct Vertex {
		float x, y, z;
		float u, v;
		uint32_t rgba;
	};

	void CreateDeviceObjects();
	void DestroyDeviceObjects();

	void DestroyPostShader();
	void DestroyStereoShader();

	static void ShowPostShaderError(const std::string &errorString);

	Draw::ShaderModule *CompileShaderModule(ShaderStage stage, ShaderLanguage lang, const std::string &src, std::string *errorString) const;
	Draw::Pipeline *CreatePipeline(std::vector<Draw::ShaderModule *> shaders, bool postShader, const UniformBufferDesc *uniformDesc) const;
	bool CompilePostShader(const ShaderInfo *shaderInfo, Draw::Pipeline **outPipeline) const;
	bool BuildPostShader(const DisplayLayoutConfig &config, const ShaderInfo *shaderInfo, const ShaderInfo *next, Draw::Pipeline **outPipeline);
	bool AllocateFramebuffer(int w, int h);

	bool BindSource(int binding, bool bindStereo);

	void GetCardboardSettings(const DisplayLayoutConfig &config, CardboardSettings *cardboardSettings) const;
	void CalculatePostShaderUniforms(int bufferWidth, int bufferHeight, int targetWidth, int targetHeight, const ShaderInfo *shaderInfo, PostShaderUniforms *uniforms) const;

	int spatialOutputWidth_ = 0;
	int spatialOutputHeight_ = 0;
	FRect sourceUV_{};
	Vertex outputVerts_[4]{};
};
