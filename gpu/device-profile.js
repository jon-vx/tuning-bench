export async function getWebGpuProfile() {
  const uaData = navigator.userAgentData
    ? {
        brands: navigator.userAgentData.brands ?? [],
        mobile: navigator.userAgentData.mobile ?? null,
        platform: navigator.userAgentData.platform ?? "",
      }
    : null;

  const profile = {
    userAgent: navigator.userAgent,
    userAgentData: uaData,
    platform: navigator.platform ?? "",
    cores: navigator.hardwareConcurrency ?? null,
    memGB: navigator.deviceMemory ?? null,
    deviceClass: inferDeviceClass(uaData),
    memory: getMemoryInfo(),
    webgpuAvailable: Boolean(navigator.gpu),
    gpu: null,
  };

  if (!navigator.gpu) return profile;

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (error) {
    profile.gpu = { error: error.message || String(error) };
    return profile;
  }
  if (!adapter) {
    profile.gpu = { error: "No WebGPU adapter" };
    return profile;
  }

  const webglInfo = getWebGlRendererInfo();
  const info =
    adapter.info ??
    (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  const normalized = normalizeGpuInfo(info, webglInfo);

  profile.gpu = {
    vendor: normalized.vendor,
    architecture: normalized.architecture,
    device: normalized.device,
    description: normalized.description,
    identity: normalized.identity,
    identitySource: normalized.identitySource,
    adapterVendor: info.vendor ?? "",
    adapterArchitecture: info.architecture ?? "",
    adapterDevice: info.device ?? "",
    adapterDescription: info.description ?? "",
    webglVendor: webglInfo.vendor,
    webglRenderer: webglInfo.renderer,
    features: [...adapter.features],
    limits: collectLimits(adapter.limits),
  };

  return profile;
}

function inferDeviceClass(uaData) {
  if (uaData?.mobile) return "phone";
  const ua = navigator.userAgent.toLowerCase();
  if (/ipad|tablet/.test(ua)) return "tablet";
  if (/mobi|android|iphone/.test(ua)) return "phone";
  return "desktop";
}

function getMemoryInfo() {
  if (!performance.memory) return null;
  return {
    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit ?? null,
    totalJSHeapSize: performance.memory.totalJSHeapSize ?? null,
    usedJSHeapSize: performance.memory.usedJSHeapSize ?? null,
  };
}

function collectLimits(limits) {
  const names = [
    "maxTextureDimension1D",
    "maxTextureDimension2D",
    "maxTextureDimension3D",
    "maxTextureArrayLayers",
    "maxBindGroups",
    "maxBindGroupsPlusVertexBuffers",
    "maxBindingsPerBindGroup",
    "maxDynamicUniformBuffersPerPipelineLayout",
    "maxDynamicStorageBuffersPerPipelineLayout",
    "maxSampledTexturesPerShaderStage",
    "maxSamplersPerShaderStage",
    "maxStorageBuffersPerShaderStage",
    "maxStorageTexturesPerShaderStage",
    "maxUniformBuffersPerShaderStage",
    "maxUniformBufferBindingSize",
    "maxStorageBufferBindingSize",
    "minUniformBufferOffsetAlignment",
    "minStorageBufferOffsetAlignment",
    "maxVertexBuffers",
    "maxBufferSize",
    "maxVertexAttributes",
    "maxVertexBufferArrayStride",
    "maxInterStageShaderVariables",
    "maxColorAttachments",
    "maxColorAttachmentBytesPerSample",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
  ];
  return Object.fromEntries(names.map((name) => [name, limits?.[name] ?? null]));
}

export function makeDefaultLabel(profile) {
  const parts = [
    profile?.deviceClass,
    profile?.platform,
    browserLabel(profile),
    profile?.gpu?.vendor,
    profile?.gpu?.architecture || profile?.gpu?.device || profile?.gpu?.identity,
  ];
  return parts.filter(Boolean).map(slugPart).join("-");
}

function browserLabel(profile) {
  const brands = profile?.userAgentData?.brands ?? [];
  const preferred = brands.find((brand) => /Chrome|Edge|Chromium/i.test(brand.brand));
  if (preferred) return `${preferred.brand}-${preferred.version}`;

  const ua = profile?.userAgent ?? "";
  const edge = /Edg\/([\d.]+)/.exec(ua);
  if (edge) return `edge-${edge[1]}`;
  const chrome = /Chrome\/([\d.]+)/.exec(ua);
  if (chrome) return `chrome-${chrome[1]}`;
  const firefox = /Firefox\/([\d.]+)/.exec(ua);
  if (firefox) return `firefox-${firefox[1]}`;
  const safari = /Version\/([\d.]+).*Safari/.exec(ua);
  if (safari) return `safari-${safari[1]}`;
  return "";
}

function slugPart(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function getWebGlRendererInfo() {
  const canvas = document.createElement("canvas");
  const gl =
    canvas.getContext("webgl") ||
    canvas.getContext("experimental-webgl") ||
    canvas.getContext("webgl2");
  if (!gl) return { vendor: "", renderer: "" };

  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
  if (debugInfo) {
    return {
      vendor: gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) ?? "",
      renderer: gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) ?? "",
    };
  }

  return {
    vendor: gl.getParameter(gl.VENDOR) ?? "",
    renderer: gl.getParameter(gl.RENDERER) ?? "",
  };
}

export function normalizeGpuInfo(info, webglInfo) {
  const identityCandidates = [
    ["adapter.description", cleanGpuString(info.description)],
    ["adapter.device", cleanGpuString(info.device)],
    ["webgl.renderer", cleanGpuString(webglInfo.renderer)],
    ["webgl.vendor", cleanGpuString(webglInfo.vendor)],
  ];
  const identityMatch = identityCandidates.find(([, value]) => value);
  const vendor = firstNonEmpty(info.vendor, inferVendorFromStrings(webglInfo.vendor, webglInfo.renderer));
  const architecture = firstNonEmpty(info.architecture, inferArchitectureFromStrings(webglInfo.renderer));
  const device = firstNonEmpty(
    cleanGpuString(info.device),
    cleanGpuString(info.description),
    extractDeviceFromRenderer(webglInfo.renderer)
  );
  const description = firstNonEmpty(
    cleanGpuString(info.description),
    cleanGpuString(info.device),
    cleanGpuString(webglInfo.renderer)
  );
  const identity = identityMatch?.[1] ?? "";
  const identitySource = identityMatch?.[0] ?? "";

  return { vendor, architecture, device, description, identity, identitySource };
}

function firstNonEmpty(...values) {
  return values.find((value) => typeof value === "string" && value.trim())?.trim() ?? "";
}

function cleanGpuString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function extractDeviceFromRenderer(renderer) {
  const text = cleanGpuString(renderer);
  if (!text) return "";
  const match = /(?:ANGLE\s*\(|Mesa\/Intel\(|Mesa\s+)([^,)]+)|\(([^)]+)\)/i.exec(text);
  if (match?.[1]) return match[1].trim();
  if (match?.[2]) return match[2].trim();
  return text;
}

function inferVendorFromStrings(...values) {
  const haystack = values.join(" ").toLowerCase();
  if (haystack.includes("intel")) return "intel";
  if (haystack.includes("nvidia") || haystack.includes("geforce") || haystack.includes("quadro")) {
    return "nvidia";
  }
  if (haystack.includes("amd") || haystack.includes("radeon")) return "amd";
  if (haystack.includes("apple")) return "apple";
  if (haystack.includes("qualcomm") || haystack.includes("adreno")) return "qualcomm";
  if (haystack.includes("arm") || haystack.includes("mali")) return "arm";
  return "";
}

function inferArchitectureFromStrings(...values) {
  const haystack = values.join(" ").toLowerCase();
  if (haystack.includes("xe lpg") || haystack.includes("xe-lpg")) return "xe-lpg";
  if (haystack.includes("xe hp") || haystack.includes("xe-hp")) return "xe-hp";
  if (haystack.includes("arc")) return "arc";
  if (haystack.includes("rdna 3") || haystack.includes("gfx11")) return "rdna3";
  if (haystack.includes("rdna 2") || haystack.includes("gfx10")) return "rdna2";
  if (haystack.includes("turing")) return "turing";
  if (haystack.includes("ampere")) return "ampere";
  if (haystack.includes("ada")) return "ada";
  if (haystack.includes("adreno")) return "adreno";
  if (haystack.includes("mali")) return "mali";
  if (haystack.includes("apple m1")) return "m1";
  if (haystack.includes("apple m2")) return "m2";
  if (haystack.includes("apple m3")) return "m3";
  return "";
}
