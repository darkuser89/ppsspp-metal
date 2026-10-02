#include "ppsspp_config.h"
#include <algorithm>
#include <array>
#include <memory>

#include "Common/StringUtils.h"

#include "GPU/Common/ShaderId.h"
#include "GPU/Common/ShaderCommon.h"
#include "GPU/Common/GPUStateUtils.h"
#include "Common/Data/Random/Rng.h"

#include "GPU/Vulkan/VulkanContext.h"

#include "GPU/Common/FragmentShaderGenerator.h"
#include "GPU/Common/VertexShaderGenerator.h"
#include "GPU/Common/ReinterpretFramebuffer.h"
#include "GPU/Common/StencilCommon.h"
#include "GPU/Common/DepalettizeShaderCommon.h"

#if PPSSPP_PLATFORM(MAC) && defined(PPSSPP_HAS_METAL)
#include "Common/GPU/Metal/MetalShaderCompiler.h"
#include "Common/GPU/Metal/thin3d_metal.h"
#include "GPU/Common/Draw2D.h"

Draw2DPipelineInfo GenerateDraw2DCopyColorFs(ShaderWriter &writer);
Draw2DPipelineInfo GenerateDraw2DCopyDepthFs(ShaderWriter &writer);
#endif

#include "UnitTest.h"

#if PPSSPP_PLATFORM(WINDOWS)
#include <wrl/client.h>
#include "GPU/D3D11/D3D11Util.h"
#include "GPU/D3D11/D3D11Loader.h"
#endif

static constexpr size_t CODE_BUFFER_SIZE = 32768;

bool GenerateFShader(FShaderID id, char *buffer, ShaderLanguage lang, Draw::Bugs bugs, std::string *errorString) {
	buffer[0] = '\0';

	FragmentShaderFlags flags;

	uint64_t uniformMask;
	switch (lang) {
	case ShaderLanguage::GLSL_VULKAN:
	{
		ShaderLanguageDesc compat(ShaderLanguage::GLSL_VULKAN);
		return GenerateFragmentShader(id, buffer, compat, bugs, &uniformMask, &flags, errorString);
	}
	case ShaderLanguage::GLSL_1xx:
	{
		ShaderLanguageDesc compat(ShaderLanguage::GLSL_1xx);
		return GenerateFragmentShader(id, buffer, compat, bugs, &uniformMask, &flags, errorString);
	}
	case ShaderLanguage::GLSL_3xx:
	{
		ShaderLanguageDesc compat(ShaderLanguage::GLSL_3xx);
		return GenerateFragmentShader(id, buffer, compat, bugs, &uniformMask, &flags, errorString);
	}
	case ShaderLanguage::HLSL_D3D11:
	{
		ShaderLanguageDesc compat(ShaderLanguage::HLSL_D3D11);
		return GenerateFragmentShader(id, buffer, compat, bugs, &uniformMask, &flags, errorString);
	}
	default:
		return false;
	}
}

bool GenerateVShader(VShaderID id, char *buffer, ShaderLanguage lang, Draw::Bugs bugs, std::string *errorString) {
	buffer[0] = '\0';

	VertexShaderFlags flags;

	uint32_t attrMask;
	uint64_t uniformMask;
	switch (lang) {
	case ShaderLanguage::GLSL_VULKAN:
	{
		ShaderLanguageDesc compat(ShaderLanguage::GLSL_VULKAN);
		return GenerateVertexShader(id, buffer, compat, bugs, &attrMask, &uniformMask, &flags, errorString);
	}
	case ShaderLanguage::GLSL_1xx:
	{
		ShaderLanguageDesc compat(ShaderLanguage::GLSL_1xx);
		return GenerateVertexShader(id, buffer, compat, bugs, &attrMask, &uniformMask, &flags, errorString);
	}
	case ShaderLanguage::GLSL_3xx:
	{
		ShaderLanguageDesc compat(ShaderLanguage::GLSL_3xx);
		return GenerateVertexShader(id, buffer, compat, bugs, &attrMask, &uniformMask, &flags, errorString);
	}
	case ShaderLanguage::HLSL_D3D11:
	{
		ShaderLanguageDesc compat(ShaderLanguage::HLSL_D3D11);
		return GenerateVertexShader(id, buffer, compat, bugs, &attrMask, &uniformMask, &flags, errorString);
	}
	default:
		return false;
	}
}

static VkShaderStageFlagBits StageToVulkan(ShaderStage stage) {
	switch (stage) {
	case ShaderStage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
	case ShaderStage::Compute: return VK_SHADER_STAGE_COMPUTE_BIT;
	case ShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	return VK_SHADER_STAGE_FRAGMENT_BIT;
}

bool TestCompileShader(const char *buffer, ShaderLanguage lang, ShaderStage stage, std::string *errorMessage) {
	std::vector<uint32_t> spirv;
	switch (lang) {
#if PPSSPP_PLATFORM(WINDOWS)
	case ShaderLanguage::HLSL_D3D11:
	{
		const char *programType = nullptr;
		switch (stage) {
		case ShaderStage::Vertex: programType = "vs_4_0"; break;
		case ShaderStage::Fragment: programType = "ps_4_0"; break;
		default:
			*errorMessage = "Unknown shader stage";
			return false;
		}
		auto output = CompileShaderToBytecodeD3D11(buffer, strlen(buffer), programType, 0, errorMessage);
		if (output.empty() && errorMessage->empty()) {
			*errorMessage = "Error compiling HLSL shader: bytecode empty";
		}
		return !output.empty();
	}
#endif

	case ShaderLanguage::GLSL_VULKAN:
		return GLSLtoSPV(StageToVulkan(stage), buffer, GLSLVariant::VULKAN, spirv, errorMessage);
	case ShaderLanguage::GLSL_1xx:
		return GLSLtoSPV(StageToVulkan(stage), buffer, GLSLVariant::GL140, spirv, errorMessage);
	case ShaderLanguage::GLSL_3xx:
		return GLSLtoSPV(StageToVulkan(stage), buffer, GLSLVariant::GLES300, spirv, errorMessage);
	default:
		*errorMessage = "Unknown shader language";
		return false;
	}
}

void PrintDiff(const char *a, const char *b) {
	// Stupidest diff ever: Just print both lines, and a few around it, when we find a mismatch.
	std::vector<std::string> a_lines;
	std::vector<std::string> b_lines;
	SplitString(a, '\n', a_lines);
	SplitString(b, '\n', b_lines);
	for (size_t i = 0; i < a_lines.size() && i < b_lines.size(); i++) {
		if (a_lines[i] != b_lines[i]) {
			// Print some context
			for (size_t j = std::max((int)i - 4, 0); j < i; j++) {
				printf("%s\n", a_lines[j].c_str());
			}
			printf("DIFF found at line %d:\n", (int)i);
			printf("a: %s\n", a_lines[i].c_str());
			printf("b: %s\n", b_lines[i].c_str());
			printf("...continues...\n");
			for (size_t j = i + 1; j < i + 5 && j < a_lines.size() && j < b_lines.size(); j++) {
				printf("a: %s\n", a_lines[j].c_str());
				printf("b: %s\n", b_lines[j].c_str());
			}
			printf("==================\n");
			return;
		}
	}
}

const char *ShaderLanguageToString(ShaderLanguage lang) {
	switch (lang) {
	case HLSL_D3D11: return "HLSL_D3D11";
	case GLSL_VULKAN: return "GLSL_VULKAN";
	case GLSL_1xx: return "GLSL_1xx";
	case GLSL_3xx: return "GLSL_3xx";
	default: return "N/A";
	}
}

bool TestReinterpretShaders() {
	Draw::Bugs bugs;

	ShaderLanguage languages[] = {
#if PPSSPP_PLATFORM(WINDOWS)
		ShaderLanguage::HLSL_D3D11,
#endif
		ShaderLanguage::GLSL_VULKAN,
		ShaderLanguage::GLSL_3xx,
	};
	GEBufferFormat fmts[3] = {
		GE_FORMAT_565,
		GE_FORMAT_5551,
		GE_FORMAT_4444,
	};
	char *buffer = new char[65536];

	// Generate all despite failures - it's only 6.
	bool failed = false;

	for (int k = 0; k < ARRAY_SIZE(languages); k++) {
		if (g_testLog) {
			printf("=== %s ===\n\n", ShaderLanguageToString(languages[k]));
		}

		ShaderLanguageDesc desc(languages[k]);

		std::string errorMessage;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == j)
					continue;  // useless shader!
				ShaderWriter writer(buffer, desc, ShaderStage::Fragment);
				GenerateReinterpretFragmentShader(writer, fmts[i], fmts[j]);
				if (strlen(buffer) >= 8192) {
					printf("Reinterpret fragment shader %d exceeded buffer:\n\n%s\n", (int)j, LineNumberString(buffer).c_str());
					failed = true;
				}
				if (!TestCompileShader(buffer, languages[k], ShaderStage::Fragment, &errorMessage)) {
					printf("Error compiling reinterpret fragment shader %d:\n\n%s\n\n%s\n", (int)j, LineNumberString(buffer).c_str(), errorMessage.c_str());
					failed = true;
				} else {
					if (g_testLog) {
						printf("===\n%s\n===\n", buffer);
					}
				}
			}
		}
	}

	delete[] buffer;
	return !failed;
}

bool TestStencilShaders() {
	Draw::Bugs bugs;

	ShaderLanguage languages[] = {
#if PPSSPP_PLATFORM(WINDOWS)
		ShaderLanguage::HLSL_D3D11,
#endif
		ShaderLanguage::GLSL_VULKAN,
		ShaderLanguage::GLSL_3xx,
	};

	char *buffer = new char[65536];

	bool failed = false;

	for (int k = 0; k < ARRAY_SIZE(languages); k++) {
		if (g_testLog) {
			printf("=== %s ===\n\n", ShaderLanguageToString(languages[k]));
		}

		ShaderLanguageDesc desc(languages[k]);
		std::string errorMessage;

		// Generate all despite failures - it's only a few.
		// Only use export on Vulkan, because GLSL_3xx is ES which doesn't support stencil export.
		bool allowUseExport = languages[k] == ShaderLanguage::GLSL_VULKAN;
		for (int useExport = 0; useExport <= (allowUseExport ? 1 : 0); ++useExport) {
			GenerateStencilFs(buffer, desc, bugs, useExport == 1);
			if (strlen(buffer) >= 8192) {
				printf("Stencil fragment shader (useExport=%d) exceeded buffer:\n\n%s\n", useExport, LineNumberString(buffer).c_str());
				failed = true;
			}
			if (!TestCompileShader(buffer, languages[k], ShaderStage::Fragment, &errorMessage)) {
				printf("Error compiling stencil shader (useExport=%d):\n\n%s\n\n%s\n", useExport, LineNumberString(buffer).c_str(), errorMessage.c_str());
				failed = true;
			} else {
				if (g_testLog) {
					printf("===\n%s\n===\n", buffer);
				}
			}
#if PPSSPP_PLATFORM(MAC) && defined(PPSSPP_HAS_METAL)
			if (useExport) {
				for (bool ios : {false, true}) {
					Metal::ShaderCompileOptions options;
					options.ios = ios;
					Metal::CompiledShader compiled;
					if (!Metal::CompileShader(buffer, ShaderStage::Fragment, options, &compiled, &errorMessage) ||
						compiled.source.find("[[stencil]]") == std::string::npos) {
						printf("Metal stencil export translation failed (%s): %s\n", ios ? "iOS" : "macOS",
							errorMessage.empty() ? "stencil output missing" : errorMessage.c_str());
						failed = true;
					}
				}
			}
#endif
		}

		GenerateStencilVs(buffer, desc);
		if (strlen(buffer) >= 8192) {
			printf("Stencil vertex shader exceeded buffer:\n\n%s\n", LineNumberString(buffer).c_str());
			failed = true;
		}
		if (!TestCompileShader(buffer, languages[k], ShaderStage::Vertex, &errorMessage)) {
			printf("Error compiling stencil shader:\n\n%s\n\n%s\n", LineNumberString(buffer).c_str(), errorMessage.c_str());
			failed = true;
		} else {
			if (g_testLog) {
				printf("===\n%s\n===\n", buffer);
			}
		}
	}

	delete[] buffer;
	return !failed;
}

#if PPSSPP_PLATFORM(MAC) && defined(PPSSPP_HAS_METAL)
static bool TestMetalShaderBlendCopyShader() {
	FShaderID id;
	id.SetBits(FS_BIT_REPLACE_BLEND, 3, REPLACE_BLEND_READ_FRAMEBUFFER);
	char buffer[65536];
	std::string error;
	if (!GenerateFShader(id, buffer, ShaderLanguage::GLSL_VULKAN, {}, &error)) {
		printf("Metal shader blend generation failed: %s\n", error.c_str());
		return false;
	}
	for (bool ios : {false, true}) {
		Metal::ShaderCompileOptions options;
		options.ios = ios;
		options.textureBindingBase = 0;
		Metal::CompiledShader compiled;
		if (!Metal::CompileShader(buffer, ShaderStage::Fragment, options, &compiled, &error)) {
			printf("Metal shader blend translation failed (%s): %s\n", ios ? "iOS" : "macOS", error.c_str());
			return false;
		}
		bool hasFramebufferTexture = std::any_of(compiled.resources.begin(), compiled.resources.end(), [](const auto &resource) {
			return resource.kind == Metal::ResourceKind::SampledTexture && resource.binding == DRAW_BINDING_2ND_TEXTURE &&
				resource.index == DRAW_BINDING_2ND_TEXTURE;
		});
		if (!hasFramebufferTexture) {
			printf("Metal shader blend framebuffer binding missing (%s)\n", ios ? "iOS" : "macOS");
			return false;
		}
	}
	return true;
}

static bool TestMetalShaderBlendFetchShader() {
	FShaderID id;
	id.SetBits(FS_BIT_REPLACE_BLEND, 3, REPLACE_BLEND_READ_FRAMEBUFFER);
	id.SetBit(FS_BIT_USE_FRAMEBUFFER_FETCH);
	char buffer[65536];
	std::string error;
	if (!GenerateFShader(id, buffer, ShaderLanguage::GLSL_VULKAN, {}, &error)) {
		printf("Metal shader blend fetch generation failed: %s\n", error.c_str());
		return false;
	}
	for (bool ios : {false, true}) {
		Metal::ShaderCompileOptions options;
		options.ios = ios;
		options.textureBindingBase = 0;
		options.framebufferFetch = true;
		Metal::CompiledShader compiled;
		if (!Metal::CompileShader(buffer, ShaderStage::Fragment, options, &compiled, &error) ||
			compiled.source.find("[[color(0)]]") == std::string::npos ||
			std::any_of(compiled.resources.begin(), compiled.resources.end(), [](const auto &resource) {
				return resource.kind == Metal::ResourceKind::SampledTexture && resource.binding == DRAW_BINDING_2ND_TEXTURE;
			})) {
			printf("Metal shader blend fetch translation failed (%s): %s\n", ios ? "iOS" : "macOS",
				error.empty() ? "color attachment input missing or bound as texture" : error.c_str());
			return false;
		}
	}
	return true;
}
#endif

bool TestDepalShaders() {
	Draw::Bugs bugs;

	ShaderLanguage languages[] = {
#if PPSSPP_PLATFORM(WINDOWS)
		ShaderLanguage::HLSL_D3D11,
#endif
		ShaderLanguage::GLSL_VULKAN,
		ShaderLanguage::GLSL_3xx,
		ShaderLanguage::GLSL_1xx,
	};

	char *buffer = new char[65536];

	for (int k = 0; k < ARRAY_SIZE(languages); k++) {
		if (g_testLog) {
			printf("=== %s ===\n\n", ShaderLanguageToString(languages[k]));
		}

		ShaderLanguageDesc desc(languages[k]);
		std::string errorMessage;

		// TODO: Try some different configurations of the fragment shader.
		// But first just try one.
		DepalConfig config{};
		config.clutFormat = GE_CMODE_16BIT_ABGR4444;
		config.shift = 8;
		config.startPos = 64;
		config.mask = 0xFF;
		config.bufferFormat = GE_FORMAT_8888;
		config.textureFormat = GE_TFMT_CLUT32;
		config.depthUpperBits = 0;

		ShaderWriter writer(buffer, desc, ShaderStage::Fragment);
		GenerateDepalFs(writer, config);
		if (strlen(buffer) >= 8192) {
			printf("Depal shader exceeded buffer:\n\n%s\n", LineNumberString(buffer).c_str());
			delete[] buffer;
			return false;
		}
		if (!TestCompileShader(buffer, languages[k], ShaderStage::Fragment, &errorMessage)) {
			printf("Error compiling depal shader:\n\n%s\n\n%s\n", LineNumberString(buffer).c_str(), errorMessage.c_str());
			delete[] buffer;
			return false;
		} else {
			if (g_testLog) {
				printf("===\n%s\n===\n", buffer);
			}
		}
	}

	delete[] buffer;
	return true;
}

const ShaderLanguage languages[] = {
#if PPSSPP_PLATFORM(WINDOWS)
	ShaderLanguage::HLSL_D3D11,
#endif
	ShaderLanguage::GLSL_VULKAN,
	ShaderLanguage::GLSL_1xx,
	ShaderLanguage::GLSL_3xx,
};
const int numLanguages = ARRAY_SIZE(languages);

bool TestVertexShaders() {
	char *buffer[numLanguages];

	for (int i = 0; i < numLanguages; i++) {
		buffer[i] = new char[65536];
	}
	GMRng rng;
	int successes = 0;
	int count = 700;

	Draw::Bugs bugs;

	// Generate a bunch of random vertex shader IDs, try to generate shader source.
	// Then compile it and check that it's ok.
	for (int i = 0; i < count; i++) {
		uint64_t id64 = rng.R64();
		VShaderID id;
		id.FromUint64(id64);

		// The generated bits need some adjustment:

		// If mode is through, we won't do hardware transform.
		if (id.Bit(VS_BIT_IS_THROUGH)) {
			id.SetBit(VS_BIT_USE_HW_TRANSFORM, 0);
		}
		if (id.Bit(VS_BIT_VERTEX_RANGE_CULLING)) {
			continue;
		}

		bool generateSuccess[numLanguages]{};
		std::string genErrorString[numLanguages];

		for (int j = 0; j < numLanguages; j++) {
			generateSuccess[j] = GenerateVShader(id, buffer[j], languages[j], bugs, &genErrorString[j]);
			if (!genErrorString[j].empty()) {
				if (g_testLog) {
					printf("%s\n", genErrorString[j].c_str());
				}
			}
		}

		for (int j = 0; j < numLanguages; j++) {
			if (strlen(buffer[j]) >= CODE_BUFFER_SIZE) {
				printf("Vertex shader exceeded buffer:\n\n%s\n", LineNumberString(buffer[j]).c_str());
				for (int i = 0; i < numLanguages; i++) {
					delete[] buffer[i];
				}
				return false;
			}
		}

		// Now that we have the strings ready for easy comparison (buffer,4 in the watch window),
		// let's try to compile them.
		for (int j = 0; j < numLanguages; j++) {
			if (generateSuccess[j]) {
				std::string errorMessage;
				if (!TestCompileShader(buffer[j], languages[j], ShaderStage::Vertex, &errorMessage)) {
					printf("Error compiling vertex shader %d:\n\n%s\n\nERROR: %s\n\n(end of error)\n\n", (int)j, LineNumberString(buffer[j]).c_str(), errorMessage.c_str());
					for (int i = 0; i < numLanguages; i++) {
						delete[] buffer[i];
					}
					return false;
				}
				successes++;
			}
		}
	}

	if (g_testLog) {
		printf("%d/%d vertex shaders generated (it's normal that it's not all, there are invalid bit combos)\n", successes, count * numLanguages);
	}

	for (int i = 0; i < numLanguages; i++) {
		delete[] buffer[i];
	}
	return true;
}

bool TestFragmentShaders() {
	char *buffer[numLanguages];

	for (int i = 0; i < numLanguages; i++) {
		buffer[i] = new char[65536];
	}
	GMRng rng;
	int successes = 0;
	int count = 300;

	Draw::Bugs bugs;

	// Generate a bunch of random fragment shader IDs, try to generate shader source.
	// Then compile it and check that it's ok.
	for (int i = 0; i < count; i++) {
		uint64_t id64 = rng.R64();
		FShaderID id;
		id.FromUint64(id64);

		// bits we don't need to test because they are irrelevant on d3d11
		id.SetBit(FS_BIT_NO_DEPTH_CANNOT_DISCARD_STENCIL, false);

		// DX9 disabling:
		if (static_cast<ReplaceAlphaType>(id.Bits(FS_BIT_STENCIL_TO_ALPHA, 2)) == ReplaceAlphaType::REPLACE_ALPHA_DUALSOURCE)
			continue;

		bool generateSuccess[numLanguages]{};
		std::string genErrorString[numLanguages];

		for (int j = 0; j < numLanguages; j++) {
			generateSuccess[j] = GenerateFShader(id, buffer[j], languages[j], bugs, &genErrorString[j]);
			if (!genErrorString[j].empty()) {
				if (g_testLog) {
					printf("%s\n", genErrorString[j].c_str());
				}
			}
			// We ignore the contents of the error string here, not even gonna try to compile if it errors.
		}

		for (int j = 0; j < numLanguages; j++) {
			if (strlen(buffer[j]) >= CODE_BUFFER_SIZE) {
				printf("Fragment shader exceeded buffer:\n\n%s\n", LineNumberString(buffer[j]).c_str());
				for (int i = 0; i < numLanguages; i++) {
					delete[] buffer[i];
				}
				return false;
			}
		}

		// Now that we have the strings ready for easy comparison (buffer,4 in the watch window),
		// let's try to compile them.
		for (int j = 0; j < numLanguages; j++) {
			if (generateSuccess[j]) {
				std::string errorMessage;
				if (!TestCompileShader(buffer[j], languages[j], ShaderStage::Fragment, &errorMessage)) {
					printf("Error compiling fragment shader %d:\n\n%s\n\nERROR: %s\n\n(end of error)\n\n", (int)j, LineNumberString(buffer[j]).c_str(), errorMessage.c_str());
					for (int i = 0; i < numLanguages; i++) {
						delete[] buffer[i];
					}
					return false;
				}
				successes++;
			}
		}
	}

	if (g_testLog) {
		printf("%d/%d fragment shaders generated (it's normal that it's not all, there are invalid bit combos)\n", successes, count * numLanguages);
	}

	for (int i = 0; i < numLanguages; i++) {
		delete[] buffer[i];
	}
	return true;
}

#if PPSSPP_PLATFORM(MAC) && defined(PPSSPP_HAS_METAL)
bool TestMetalDelayedReadback() {
	std::string error;
	std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
	if (!draw) {
		if (error == "This device does not support Metal 3" || error == "Metal 3 requires macOS 13 or iOS 16 or later") {
			printf("Metal delayed readback skipped: %s\n", error.c_str());
			return true;
		}
		printf("Metal context failed: %s\n", error.c_str());
		return false;
	}
	Draw::AutoRef<Draw::Framebuffer> framebuffer(draw->CreateFramebuffer({4, 2, 1, 1, 0, false, "Metal delayed readback test"}));
	if (!framebuffer) {
		printf("Metal framebuffer creation failed\n");
		return false;
	}
	constexpr uint32_t untouched = 0x12345678;
	std::array<uint32_t, 12> pixels;
	auto check = [&](const char *stage, uint32_t expected) {
		for (int y = 0; y < 2; ++y) {
			for (int x = 0; x < 6; ++x) {
				uint32_t wanted = x < 4 ? expected : untouched;
				if (pixels[y * 6 + x] != wanted) {
					printf("%s pixel %d,%d: got %08x, expected %08x\n", stage, x, y, pixels[y * 6 + x], wanted);
					return false;
				}
			}
		}
		return true;
	};
	draw->BeginFrame(Draw::DebugFlags::NONE);
	draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::KEEP, Draw::RPAction::KEEP, 0xFF0000FF, 1.0f, 0, "Red"}, "Red");
	pixels.fill(untouched);
	if (draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "First") ||
		!check("First", untouched)) {
		printf("Metal first delayed readback returned data before submission\n");
		return false;
	}
	draw->Wait();
	pixels.fill(untouched);
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::B8G8R8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "Red BGRA") ||
		!check("Red BGRA", 0xFFFF0000)) {
		return false;
	}
	draw->Clear(Draw::Aspect::COLOR_BIT, 0xFFFF0000, 1.0f, 0);
	draw->Wait();
	pixels.fill(untouched);
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "Old red") ||
		!check("Old red", 0xFF0000FF)) {
		return false;
	}
	draw->Wait();
	pixels.fill(untouched);
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "New blue") ||
		!check("New blue", 0xFFFF0000)) {
		return false;
	}
	draw->Wait();
	uint32_t previousColor = 0xFFFF0000;
	for (uint32_t color : {0xFF123456u, 0xFFABCDEFu, 0xFF987654u, 0xFF102030u}) {
		draw->Clear(Draw::Aspect::COLOR_BIT, color, 1.0f, 0);
		for (int repeat = 0; repeat < 2; ++repeat) {
			pixels.fill(untouched);
			if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
				Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "Pending color") ||
				!check("Pending color", previousColor)) {
				return false;
			}
		}
		draw->Wait();
		previousColor = color;
	}
	pixels.fill(untouched);
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "Final color") ||
		!check("Final color", previousColor)) {
		return false;
	}
	std::unique_ptr<Draw::DrawContext> bounded(Draw::T3DCreateMetalContext(&error));
	if (!bounded) {
		printf("Metal bounded readback context failed: %s\n", error.c_str());
		return false;
	}
	std::array<Draw::AutoRef<Draw::Framebuffer>, 9> sources;
	for (auto &source : sources) {
		source = bounded->CreateFramebuffer({4, 2, 1, 1, 0, false, "Metal bounded readback test"});
		if (!source) {
			printf("Metal bounded readback framebuffer creation failed\n");
			return false;
		}
	}
	bounded->BeginFrame(Draw::DebugFlags::NONE);
	for (size_t i = 0; i < sources.size(); ++i) {
		const uint32_t color = 0xFF000020 + (uint32_t)i * 0x10;
		bounded->BindFramebufferAsRenderTarget(sources[i].ptr,
			{Draw::RPAction::CLEAR, Draw::RPAction::KEEP, Draw::RPAction::KEEP, color, 1.0f, 0, "Bounded readback"}, "Bounded readback");
		pixels.fill(untouched);
		const bool ready = bounded->CopyFramebufferToMemory(sources[i].ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
			Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "Bounded readback");
		if (i < sources.size() - 1) {
			if (ready || !check("Pending bounded readback", untouched)) {
				printf("Metal bounded readback returned unfinished data at source %zu\n", i);
				return false;
			}
		} else if (!ready || !check("Bounded fallback readback", color)) {
			printf("Metal bounded readback fallback failed\n");
			return false;
		}
	}
	bounded->Wait();
	pixels.fill(untouched);
	if (!bounded->CopyFramebufferToMemory(sources[0].ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 6, Draw::ReadbackMode::OLD_DATA_OK, "Completed bounded readback") ||
		!check("Completed bounded readback", 0xFF000020)) {
		return false;
	}
	return true;
}

bool TestMetalTriangleFan() {
	std::string error;
	std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
	if (!draw) {
		if (error == "This device does not support Metal 3" || error == "Metal 3 requires macOS 13 or iOS 16 or later") {
			printf("Metal triangle fan skipped: %s\n", error.c_str());
			return true;
		}
		printf("Metal triangle fan context failed: %s\n", error.c_str());
		return false;
	}
	init_glslang();
	const char *vsSource = "#version 450\n"
		"layout(location = 0) in vec2 a_position;\n"
		"void main() { gl_Position = vec4(a_position, 0.0, 1.0); }\n";
	const char *fsSource = "#version 450\n"
		"layout(location = 0) out vec4 o_color;\n"
		"void main() { o_color = vec4(1.0, 0.0, 0.0, 1.0); }\n";
	const auto language = draw->GetShaderLanguageDesc().shaderLanguage;
	Draw::AutoRef<Draw::ShaderModule> vs(draw->CreateShaderModule(ShaderStage::Vertex, language,
		(const uint8_t *)vsSource, strlen(vsSource), "Metal triangle fan VS"));
	Draw::AutoRef<Draw::ShaderModule> fs(draw->CreateShaderModule(ShaderStage::Fragment, language,
		(const uint8_t *)fsSource, strlen(fsSource), "Metal triangle fan FS"));
	Draw::AutoRef<Draw::InputLayout> input(draw->CreateInputLayout({sizeof(float) * 2,
		{{Draw::SEM_POSITION, Draw::DataFormat::R32G32_FLOAT, 0}}}));
	Draw::AutoRef<Draw::DepthStencilState> depth(draw->CreateDepthStencilState({}));
	Draw::AutoRef<Draw::BlendState> blend(draw->CreateBlendState({false, 0xF}));
	Draw::AutoRef<Draw::RasterState> raster(draw->CreateRasterState({}));
	if (!vs || !fs || !input || !depth || !blend || !raster) {
		printf("Metal triangle fan resources failed\n");
		return false;
	}
	Draw::PipelineDesc desc{Draw::Primitive::TRIANGLE_FAN, {vs.ptr, fs.ptr},
		input.ptr, depth.ptr, blend.ptr, raster.ptr, nullptr};
	Draw::AutoRef<Draw::Pipeline> pipeline(draw->CreateGraphicsPipeline(desc, "Metal triangle fan"));
	Draw::AutoRef<Draw::Framebuffer> framebuffer(draw->CreateFramebuffer({8, 8, 1, 1, 0, false, "Metal triangle fan"}));
	if (!pipeline || !framebuffer) {
		printf("Metal triangle fan pipeline or framebuffer failed\n");
		return false;
	}
	const std::array<std::array<float, 2>, 4> quad{{{-0.8f, -0.8f}, {0.8f, -0.8f},
		{0.8f, 0.8f}, {-0.8f, 0.8f}}};
	const std::array<std::array<float, 2>, 5> indexedQuad{{{2.0f, 2.0f},
		{-0.8f, -0.8f}, {0.8f, -0.8f}, {0.8f, 0.8f}, {-0.8f, 0.8f}}};
	const std::array<uint16_t, 4> indices{{1, 2, 3, 4}};
	Draw::AutoRef<Draw::Buffer> quadBuffer(draw->CreateBuffer(sizeof(quad), 0));
	Draw::AutoRef<Draw::Buffer> indexedQuadBuffer(draw->CreateBuffer(sizeof(indexedQuad), 0));
	Draw::AutoRef<Draw::Buffer> indexBuffer(draw->CreateBuffer(sizeof(indices), 0));
	if (!quadBuffer || !indexedQuadBuffer || !indexBuffer) {
		printf("Metal triangle fan buffer creation failed\n");
		return false;
	}
	draw->UpdateBuffer(quadBuffer.ptr, (const uint8_t *)quad.data(), 0, sizeof(quad), Draw::UPDATE_DISCARD);
	draw->UpdateBuffer(indexedQuadBuffer.ptr, (const uint8_t *)indexedQuad.data(), 0, sizeof(indexedQuad), Draw::UPDATE_DISCARD);
	draw->UpdateBuffer(indexBuffer.ptr, (const uint8_t *)indices.data(), 0, sizeof(indices), Draw::UPDATE_DISCARD);
	std::array<uint32_t, 64> pixels{};
	draw->BeginFrame(Draw::DebugFlags::NONE);
	draw->BindPipeline(pipeline.ptr);
	draw->SetViewport({0.0f, 0.0f, 8.0f, 8.0f, 0.0f, 1.0f});
	draw->SetScissorRect(0, 0, 8, 8);
	for (int mode = 0; mode < 4; ++mode) {
		draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
			{Draw::RPAction::CLEAR, Draw::RPAction::KEEP, Draw::RPAction::KEEP, 0xFFFF0000, 1.0f, 0,
				"Metal triangle fan clear"}, "Metal triangle fan clear");
		switch (mode) {
		case 0:
			draw->DrawUP(quad.data(), (int)quad.size());
			break;
		case 1:
			draw->DrawIndexedUP(indexedQuad.data(), (int)indexedQuad.size(), indices.data(), (int)indices.size());
			break;
		case 2:
			draw->BindVertexBuffer(quadBuffer.ptr, 0);
			draw->Draw((int)quad.size(), 0);
			break;
		case 3:
			draw->BindVertexBuffer(indexedQuadBuffer.ptr, 0);
			draw->BindIndexBuffer(indexBuffer.ptr, 0);
			draw->DrawIndexed((int)indices.size(), 0);
			break;
		}
		if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 8, 8,
			Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 8, Draw::ReadbackMode::BLOCK, "Metal triangle fan readback") ||
			pixels[0] != 0xFFFF0000 || pixels[2 * 8 + 5] != 0xFF0000FF ||
			pixels[5 * 8 + 2] != 0xFF0000FF) {
			printf("Metal triangle fan mode %d pixels: %08x %08x %08x\n", mode,
				pixels[0], pixels[2 * 8 + 5], pixels[5 * 8 + 2]);
			return false;
		}
	}
	return true;
}

bool TestMetalCullPointsLines() {
	std::string error;
	std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
	if (!draw) {
		if (error == "This device does not support Metal 3" || error == "Metal 3 requires macOS 13 or iOS 16 or later") {
			printf("Metal point/line culling skipped: %s\n", error.c_str());
			return true;
		}
		printf("Metal point/line culling context failed: %s\n", error.c_str());
		return false;
	}
	init_glslang();
	const char *vsSource = "#version 450\n"
		"layout(location = 0) in vec2 a_position;\n"
		"void main() { gl_Position = vec4(a_position, 0.0, 1.0); }\n";
	const char *pointVsSource = "#version 450\n"
		"layout(location = 0) in vec2 a_position;\n"
		"void main() { gl_Position = vec4(a_position, 0.0, 1.0); gl_PointSize = 1.0; }\n";
	const char *fsSource = "#version 450\n"
		"layout(location = 0) out vec4 o_color;\n"
		"void main() { o_color = vec4(1.0, 0.0, 0.0, 1.0); }\n";
	const auto language = draw->GetShaderLanguageDesc().shaderLanguage;
	Draw::AutoRef<Draw::ShaderModule> vs(draw->CreateShaderModule(ShaderStage::Vertex, language,
		(const uint8_t *)vsSource, strlen(vsSource), "Metal point/line culling VS"));
	Draw::AutoRef<Draw::ShaderModule> pointVs(draw->CreateShaderModule(ShaderStage::Vertex, language,
		(const uint8_t *)pointVsSource, strlen(pointVsSource), "Metal point culling VS"));
	Draw::AutoRef<Draw::ShaderModule> fs(draw->CreateShaderModule(ShaderStage::Fragment, language,
		(const uint8_t *)fsSource, strlen(fsSource), "Metal point/line culling FS"));
	Draw::AutoRef<Draw::InputLayout> input(draw->CreateInputLayout({sizeof(float) * 2,
		{{Draw::SEM_POSITION, Draw::DataFormat::R32G32_FLOAT, 0}}}));
	Draw::AutoRef<Draw::DepthStencilState> depth(draw->CreateDepthStencilState({}));
	Draw::AutoRef<Draw::BlendState> blend(draw->CreateBlendState({false, 0xF}));
	Draw::AutoRef<Draw::RasterState> raster(draw->CreateRasterState({Draw::CullMode::FRONT_AND_BACK, Draw::Facing::CCW}));
	Draw::AutoRef<Draw::RasterState> noCull(draw->CreateRasterState({Draw::CullMode::NONE, Draw::Facing::CCW}));
	Draw::AutoRef<Draw::Framebuffer> framebuffer(draw->CreateFramebuffer({8, 8, 1, 1, 0, false, "Metal point/line culling"}));
	if (!vs || !pointVs || !fs || !input || !depth || !blend || !raster || !noCull || !framebuffer) {
		printf("Metal point/line culling resources failed\n");
		return false;
	}
	const std::array<Draw::Primitive, 4> primitives{{Draw::Primitive::POINT_LIST,
		Draw::Primitive::LINE_LIST, Draw::Primitive::LINE_LIST, Draw::Primitive::TRIANGLE_LIST}};
	const std::array<std::array<float, 2>, 3> vertices{{{-0.75f, -0.75f}, {0.75f, 0.75f}, {-0.75f, 0.75f}}};
	std::array<uint32_t, 64> pixels{};
	std::array<uint32_t, 64> lineWithoutCulling{};
	draw->BeginFrame(Draw::DebugFlags::NONE);
	draw->SetViewport({0.0f, 0.0f, 8.0f, 8.0f, 0.0f, 1.0f});
	draw->SetScissorRect(0, 0, 8, 8);
	for (size_t i = 0; i < primitives.size(); ++i) {
		Draw::PipelineDesc desc{primitives[i], {i == 0 ? pointVs.ptr : vs.ptr, fs.ptr},
			input.ptr, depth.ptr, blend.ptr, i == 1 ? noCull.ptr : raster.ptr, nullptr};
		Draw::AutoRef<Draw::Pipeline> pipeline(draw->CreateGraphicsPipeline(desc, "Metal point/line culling"));
		if (!pipeline) {
			printf("Metal point/line culling pipeline %zu failed\n", i);
			return false;
		}
		draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
			{Draw::RPAction::CLEAR, Draw::RPAction::KEEP, Draw::RPAction::KEEP, 0xFFFF0000, 1.0f, 0,
				"Metal point/line culling clear"}, "Metal point/line culling clear");
		draw->BindPipeline(pipeline.ptr);
		draw->DrawUP(vertices.data(), i == 0 ? 1 : i == 3 ? 3 : 2);
		if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 8, 8,
			Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 8, Draw::ReadbackMode::BLOCK, "Metal point/line culling readback")) {
			printf("Metal point/line culling readback %zu failed\n", i);
			return false;
		}
		const int drawn = (int)std::count(pixels.begin(), pixels.end(), 0xFF0000FF);
		if ((drawn == 0 && i < 3) || (drawn != 0 && i == 3)) {
			const int changed = (int)std::count_if(pixels.begin(), pixels.end(), [](uint32_t pixel) { return pixel != 0xFFFF0000; });
			printf("Metal point/line culling primitive %zu drew %d red pixels, %d changed pixels; center %08x\n",
				i, drawn, changed, pixels[3 * 8 + 3]);
			return false;
		}
		if (i == 1) {
			lineWithoutCulling = pixels;
		} else if (i == 2 && pixels != lineWithoutCulling) {
			printf("Metal line coverage changed with front-and-back culling\n");
			return false;
		}
	}
	return true;
}

bool TestMetalMSAAResolve() {
	std::string error;
	std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
	if (!draw) {
		if (error == "This device does not support Metal 3" || error == "Metal 3 requires macOS 13 or iOS 16 or later") {
			printf("Metal MSAA resolve skipped: %s\n", error.c_str());
			return true;
		}
		printf("Metal context failed: %s\n", error.c_str());
		return false;
	}
	if (!(draw->GetDeviceCaps().multiSampleLevelsMask & (1u << 1))) {
		printf("Metal MSAA resolve skipped: 2x MSAA unavailable\n");
		return true;
	}
	Draw::AutoRef<Draw::Framebuffer> framebuffer(draw->CreateFramebuffer({4, 2, 1, 1, 1, true, "Metal MSAA resolve test"}));
	if (!framebuffer) {
		printf("Metal MSAA framebuffer creation failed\n");
		return false;
	}
	draw->BeginFrame(Draw::DebugFlags::NONE);
	draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF0000FF, 0.25f, 0x5A, "MSAA clear"}, "MSAA clear");
	std::array<uint32_t, 8> color{};
	std::array<float, 8> depth{};
	std::array<uint8_t, 8> stencil{};
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, color.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA color") ||
		!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 4, 2,
		Draw::DataFormat::D32F, depth.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA depth") ||
		!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 4, 2,
		Draw::DataFormat::S8, stencil.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA stencil")) {
		printf("Metal MSAA resolve readback failed\n");
		return false;
	}
	for (size_t i = 0; i < color.size(); ++i) {
		if (color[i] != 0xFF0000FF || depth[i] != 0.25f || stencil[i] != 0x5A) {
			printf("Metal MSAA resolve pixel %zu: color %08x, depth %f, stencil %02x\n",
				i, color[i], depth[i], stencil[i]);
			return false;
		}
	}
	draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF00FF00, 0.5f, 0xA5, "MSAA discard"}, "MSAA discard");
	draw->InvalidateFramebuffer(Draw::FB_INVALIDATION_STORE, Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT);
	color.fill(0);
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, color.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA discard color")) {
		printf("Metal MSAA color resolve after depth/stencil discard failed\n");
		return false;
	}
	for (size_t i = 0; i < color.size(); ++i) {
		if (color[i] != 0xFF00FF00) {
			printf("Metal MSAA discard pixel %zu: color %08x, expected green\n", i, color[i]);
			return false;
		}
	}
	draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFFFF0000, 0.75f, 0x3C, "MSAA color discard"}, "MSAA color discard");
	draw->InvalidateFramebuffer(Draw::FB_INVALIDATION_STORE, Draw::Aspect::COLOR_BIT);
	depth.fill(0.0f);
	stencil.fill(0);
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 4, 2,
		Draw::DataFormat::D32F, depth.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA color discard depth") ||
		!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 4, 2,
		Draw::DataFormat::S8, stencil.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA color discard stencil")) {
		printf("Metal MSAA depth/stencil resolve after color discard failed\n");
		return false;
	}
	for (size_t i = 0; i < depth.size(); ++i) {
		if (depth[i] != 0.75f || stencil[i] != 0x3C) {
			printf("Metal MSAA color discard pixel %zu: depth %f, stencil %02x\n", i, depth[i], stencil[i]);
			return false;
		}
	}
	Draw::AutoRef<Draw::Framebuffer> scaled(draw->CreateFramebuffer({8, 4, 1, 1, 1, true, "Metal MSAA scaled blit target"}));
	if (!scaled) {
		printf("Metal MSAA scaled blit framebuffer creation failed\n");
		return false;
	}
	draw->BindFramebufferAsRenderTarget(framebuffer.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF00FF00, 0.5f, 0xA5, "MSAA blit source"}, "MSAA blit source");
	draw->BindFramebufferAsRenderTarget(scaled.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF000000, 1.0f, 0, "MSAA blit target"}, "MSAA blit target");
	if (!draw->BlitFramebuffer(framebuffer.ptr, 0, 0, 4, 2, scaled.ptr, 0, 0, 8, 4,
		Draw::Aspect::COLOR_BIT | Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT, Draw::FB_BLIT_NEAREST, "MSAA scaled blit")) {
		printf("Metal MSAA scaled blit failed\n");
		return false;
	}
	std::array<uint32_t, 32> scaledColor{};
	std::array<float, 32> scaledDepth{};
	std::array<uint8_t, 32> scaledStencil{};
	if (!draw->CopyFramebufferToMemory(scaled.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 8, 4,
		Draw::DataFormat::R8G8B8A8_UNORM, scaledColor.data(), 8, Draw::ReadbackMode::BLOCK, "MSAA blit color") ||
		!draw->CopyFramebufferToMemory(scaled.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 8, 4,
		Draw::DataFormat::D32F, scaledDepth.data(), 8, Draw::ReadbackMode::BLOCK, "MSAA blit depth") ||
		!draw->CopyFramebufferToMemory(scaled.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 8, 4,
		Draw::DataFormat::S8, scaledStencil.data(), 8, Draw::ReadbackMode::BLOCK, "MSAA blit stencil")) {
		printf("Metal MSAA scaled blit readback failed\n");
		return false;
	}
	for (size_t i = 0; i < scaledColor.size(); ++i) {
		if (scaledColor[i] != 0xFF00FF00 || scaledDepth[i] != 0.5f || scaledStencil[i] != 0xA5) {
			printf("Metal MSAA scaled blit pixel %zu: color %08x, depth %f, stencil %02x\n",
				i, scaledColor[i], scaledDepth[i], scaledStencil[i]);
			return false;
		}
	}
	draw->BindFramebufferAsRenderTarget(scaled.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF0000FF, 0.25f, 0x5A, "MSAA stencil blit target"}, "MSAA stencil blit target");
	if (!draw->BlitFramebuffer(framebuffer.ptr, 0, 0, 4, 2, scaled.ptr, 0, 0, 8, 4,
		Draw::Aspect::STENCIL_BIT, Draw::FB_BLIT_NEAREST, "MSAA stencil-only blit")) {
		printf("Metal MSAA stencil-only blit failed\n");
		return false;
	}
	if (!draw->CopyFramebufferToMemory(scaled.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 8, 4,
		Draw::DataFormat::R8G8B8A8_UNORM, scaledColor.data(), 8, Draw::ReadbackMode::BLOCK, "MSAA stencil blit color") ||
		!draw->CopyFramebufferToMemory(scaled.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 8, 4,
		Draw::DataFormat::D32F, scaledDepth.data(), 8, Draw::ReadbackMode::BLOCK, "MSAA stencil blit depth") ||
		!draw->CopyFramebufferToMemory(scaled.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 8, 4,
		Draw::DataFormat::S8, scaledStencil.data(), 8, Draw::ReadbackMode::BLOCK, "MSAA stencil blit stencil")) {
		printf("Metal MSAA stencil-only blit readback failed\n");
		return false;
	}
	for (size_t i = 0; i < scaledColor.size(); ++i) {
		if (scaledColor[i] != 0xFF0000FF || scaledDepth[i] != 0.25f || scaledStencil[i] != 0xA5) {
			printf("Metal MSAA stencil-only blit pixel %zu: color %08x, depth %f, stencil %02x\n",
				i, scaledColor[i], scaledDepth[i], scaledStencil[i]);
			return false;
		}
	}
	draw->BindFramebufferAsRenderTarget(scaled.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF0000FF, 0.25f, 0x5A, "MSAA self-copy source"}, "MSAA self-copy source");
	if (!draw->BlitFramebuffer(scaled.ptr, 0, 0, 8, 4, framebuffer.ptr, 0, 0, 2, 2,
		Draw::Aspect::COLOR_BIT | Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT, Draw::FB_BLIT_NEAREST, "MSAA left half")) {
		printf("Metal MSAA left-half preparation failed\n");
		return false;
	}
	if (!draw->BlitFramebuffer(framebuffer.ptr, 0, 0, 2, 2, framebuffer.ptr, 2, 0, 4, 2,
		Draw::Aspect::COLOR_BIT | Draw::Aspect::DEPTH_BIT | Draw::Aspect::STENCIL_BIT, Draw::FB_BLIT_NEAREST, "MSAA self-copy")) {
		printf("Metal MSAA self-copy failed\n");
		return false;
	}
	if (!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, color.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA self-copy color") ||
		!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 4, 2,
		Draw::DataFormat::D32F, depth.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA self-copy depth") ||
		!draw->CopyFramebufferToMemory(framebuffer.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 4, 2,
		Draw::DataFormat::S8, stencil.data(), 4, Draw::ReadbackMode::BLOCK, "MSAA self-copy stencil")) {
		printf("Metal MSAA self-copy readback failed\n");
		return false;
	}
	for (size_t i = 0; i < color.size(); ++i) {
		if (color[i] != 0xFF0000FF || depth[i] != 0.25f || stencil[i] != 0x5A) {
			printf("Metal MSAA self-copy pixel %zu: color %08x, depth %f, stencil %02x\n",
				i, color[i], depth[i], stencil[i]);
			return false;
		}
	}
	return true;
}

bool TestMetalLoadDiscardBlit() {
	init_glslang();
	std::string error;
	std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
	if (!draw) {
		if (error == "This device does not support Metal 3" || error == "Metal 3 requires macOS 13 or iOS 16 or later") {
			printf("Metal load discard blit skipped: %s\n", error.c_str());
			return true;
		}
		printf("Metal context failed: %s\n", error.c_str());
		return false;
	}
	Draw::AutoRef<Draw::Framebuffer> source(draw->CreateFramebuffer({4, 2, 1, 1, 0, true, "Metal blit source"}));
	Draw::AutoRef<Draw::Framebuffer> target(draw->CreateFramebuffer({4, 2, 1, 1, 0, true, "Metal blit target"}));
	if (!source || !target) {
		printf("Metal load discard framebuffer creation failed\n");
		return false;
	}
	auto releaseDraw2D = [](Draw2D *helper) { helper->DeviceLost(); delete helper; };
	std::unique_ptr<Draw2D, decltype(releaseDraw2D)> draw2D(new Draw2D(draw.get()), releaseDraw2D);
	auto releasePipeline = [](Draw2DPipeline *pipeline) { if (pipeline) { pipeline->Release(); } };
	std::unique_ptr<Draw2DPipeline, decltype(releasePipeline)> pipeline(draw2D->Create2DPipeline(&GenerateDraw2DCopyColorFs), releasePipeline);
	std::unique_ptr<Draw2DPipeline, decltype(releasePipeline)> depthPipeline(draw2D->Create2DPipeline(&GenerateDraw2DCopyDepthFs), releasePipeline);
	if (!pipeline || !pipeline->pipeline || !depthPipeline || !depthPipeline->pipeline) {
		printf("Metal load discard blit pipeline creation failed\n");
		return false;
	}
	draw->BeginFrame(Draw::DebugFlags::NONE);
	draw->BindFramebufferAsRenderTarget(source.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFFFF0000, 0.75f, 0xC3, "Blue source"}, "Blue source");
	draw->BindFramebufferAsRenderTarget(target.ptr,
		{Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, 0xFF0000FF, 0.25f, 0x5A, "Red target"}, "Red target");
	draw->BindFramebufferAsRenderTarget(target.ptr,
		{Draw::RPAction::KEEP, Draw::RPAction::KEEP, Draw::RPAction::KEEP}, "Load discard blit");
	draw->BindFramebufferAsTexture(source.ptr, 0, Draw::Aspect::COLOR_BIT, Draw::ALL_LAYERS);
	draw->InvalidateFramebuffer(Draw::FB_INVALIDATION_LOAD, Draw::Aspect::COLOR_BIT);
	draw->SetViewport({0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 1.0f});
	draw->SetScissorRect(0, 0, 4, 2);
	draw2D->Blit(pipeline.get(), 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.0f, 4.0f, 2.0f,
		4.0f, 2.0f, 4.0f, 2.0f, false, 1);
	std::array<uint32_t, 8> color{};
	std::array<float, 8> depth{};
	std::array<uint8_t, 8> stencil{};
	if (!draw->CopyFramebufferToMemory(target.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, color.data(), 4, Draw::ReadbackMode::BLOCK, "Metal load discard color") ||
		!draw->CopyFramebufferToMemory(target.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 4, 2,
		Draw::DataFormat::D32F, depth.data(), 4, Draw::ReadbackMode::BLOCK, "Metal load discard depth") ||
		!draw->CopyFramebufferToMemory(target.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 4, 2,
		Draw::DataFormat::S8, stencil.data(), 4, Draw::ReadbackMode::BLOCK, "Metal load discard stencil")) {
		printf("Metal load discard blit readback failed\n");
		return false;
	}
	for (size_t i = 0; i < color.size(); ++i) {
		if (color[i] != 0xFFFF0000 || depth[i] != 0.25f || stencil[i] != 0x5A) {
			printf("Metal load discard pixel %zu: color %08x, depth %f, stencil %02x\n", i, color[i], depth[i], stencil[i]);
			return false;
		}
	}
	draw->BindFramebufferAsRenderTarget(target.ptr,
		{Draw::RPAction::KEEP, Draw::RPAction::KEEP, Draw::RPAction::KEEP}, "Depth load discard blit");
	draw->BindFramebufferAsTexture(source.ptr, 0, Draw::Aspect::DEPTH_BIT, Draw::ALL_LAYERS);
	draw->InvalidateFramebuffer(Draw::FB_INVALIDATION_LOAD, Draw::Aspect::DEPTH_BIT);
	draw2D->Blit(depthPipeline.get(), 0.0f, 0.0f, 4.0f, 2.0f, 0.0f, 0.0f, 4.0f, 2.0f,
		4.0f, 2.0f, 4.0f, 2.0f, false, 1);
	if (!draw->CopyFramebufferToMemory(target.ptr, Draw::Aspect::COLOR_BIT, 0, 0, 4, 2,
		Draw::DataFormat::R8G8B8A8_UNORM, color.data(), 4, Draw::ReadbackMode::BLOCK, "Metal depth blit color") ||
		!draw->CopyFramebufferToMemory(target.ptr, Draw::Aspect::DEPTH_BIT, 0, 0, 4, 2,
		Draw::DataFormat::D32F, depth.data(), 4, Draw::ReadbackMode::BLOCK, "Metal depth blit depth") ||
		!draw->CopyFramebufferToMemory(target.ptr, Draw::Aspect::STENCIL_BIT, 0, 0, 4, 2,
		Draw::DataFormat::S8, stencil.data(), 4, Draw::ReadbackMode::BLOCK, "Metal depth blit stencil")) {
		printf("Metal depth load discard blit readback failed\n");
		return false;
	}
	for (size_t i = 0; i < color.size(); ++i) {
		if (color[i] != 0xFFFF0000 || depth[i] != 0.75f || stencil[i] != 0x5A) {
			printf("Metal depth load discard pixel %zu: color %08x, depth %f, stencil %02x\n", i, color[i], depth[i], stencil[i]);
			return false;
		}
	}
	return true;
}
#endif

bool TestShaderGenerators() {
#if PPSSPP_PLATFORM(WINDOWS)
	LoadD3D11();
	init_glslang();
#else
	init_glslang();
#endif

	if (!TestStencilShaders()) {
		return false;
	}

#if PPSSPP_PLATFORM(MAC) && defined(PPSSPP_HAS_METAL)
	if (!TestMetalShaderBlendCopyShader()) {
		return false;
	}
	if (!TestMetalShaderBlendFetchShader()) {
		return false;
	}
#endif

	if (!TestReinterpretShaders()) {
		return false;
	}

	if (!TestDepalShaders()) {
		return false;
	}

	if (!TestFragmentShaders()) {
		return false;
	}

	if (!TestVertexShaders()) {
		return false;
	}

	return true;
} 
