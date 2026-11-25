#pragma once

#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 配置项结构定义
struct SfMConfig {
	bool useFocalLength;
	double useFactor;
	std::string cameraModel;
	int numLocalBAIters;
	size_t maxNumFeatures;
	size_t maxPendingImages;
	int RegistrationRetryInterval;
	int RegistrationRetryIdleSeconds;
};

struct BundleAdjustmentRuntimeConfig {
	int LocalBACudaMinObservations;
};

struct PipelineConfig {
	bool AutoRunImages;
	bool AutoExportModel;
	int InitialImagesNum;
	bool Export3DGS;
	bool OutputTimingFiles;
};

struct ExperimentConfig {
	bool Enable;
	std::string ProjectPath;
	std::string ImagePath;
	bool ShowUI;
	int AutoSaveIdleSeconds;
};

struct PythonConfig {
	std::string PythonHome;
	std::string PythonWorkspace;
	//std::string GlobalFeatureModel;
};

struct GlobalFeatureConfig {
	std::string GlobalFeatureType; // 全局特征模型类型 (VGG / SupScene)

	// SupScene 配置参数
	std::string SupSceneDevice;        // 设备：cuda or cpu
	std::string SupSceneWeightsPath;   // 权重路径，auto 表示使用 scripts/pyscripts/Supscene/checkpoints/dinov2_scpp_supscene_1536.pth
};

struct DenseDepthConfig {
	bool Enable;
	int StartRegisteredImages;
	std::string Method;
	std::string Device;
	int TimeoutMs;
	bool ExportSfMBin;
	int OutputResolution;
	int MurreProcessingResolution;
	double MurreMaxDepth;
	int MurreDenoiseSteps;
	int MurreEnsembleSize;

	int PriorDAProcessingResolution;
	double PriorDAMaxDepth;
	std::string PriorDAVersion;
	std::string PriorDAFrozenModelSize;
	std::string PriorDAConditionedModelSize;
	std::string PriorDAPattern;

	int DepthAnythingV2ProcessingResolution;
	double DepthAnythingV2MaxDepth;
	std::string DepthAnythingV2Encoder;

	std::string DepthAnything3Model;
	int DepthAnything3ProcessingResolution;
	double DepthAnything3MaxDepth;
	std::string DepthAnything3ProcessResMethod;
	double DepthAnything3ConfThreshold;

	int PixelPerfectDepthProcessingResolution;
	int PixelPerfectDepthSamplingSteps;
	double PixelPerfectDepthMaxDepth;
	bool PixelPerfectDepthFP16;
	std::string PixelPerfectDepthSemanticsModel;

	std::string MoGeModel;
	int MoGeResolutionLevel;
	int MoGeNumTokens;
	bool MoGeUseFP16;
	bool MoGeUseSfMFov;
	bool MoGeApplyMask;
	double MoGeMaskThreshold;
	bool MoGeSaveNormal;
	double MoGeMaxDepth;

	bool IsPriorDepthAnything() const {
		std::string method = Method;
		std::transform(method.begin(), method.end(), method.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return method == "priordepthanything" ||
			method == "priorda" ||
			method == "prior-depth-anything";
	}

	bool IsDepthAnythingV2() const {
		std::string method = Method;
		std::transform(method.begin(), method.end(), method.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return method == "depthanythingv2" ||
			method == "depthanything-v2" ||
			method == "depth-anything-v2" ||
			method == "dav2";
	}

	bool IsDepthAnything3() const {
		std::string method = Method;
		std::transform(method.begin(), method.end(), method.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return method == "depthanything3" ||
			method == "depthanything-v3" ||
			method == "depth-anything-3" ||
			method == "da3";
	}

	bool IsPixelPerfectDepth() const {
		std::string method = Method;
		std::transform(method.begin(), method.end(), method.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return method == "pixelperfectdepth" ||
			method == "pixel-perfect-depth" ||
			method == "ppd";
	}

	bool IsMoGe() const {
		std::string method = Method;
		std::transform(method.begin(), method.end(), method.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return method == "moge" ||
			method == "moge2" ||
			method == "moge-2";
	}

	int ProcessingResolution() const {
		if (IsPriorDepthAnything()) {
			return PriorDAProcessingResolution;
		}
		if (IsDepthAnythingV2()) {
			return DepthAnythingV2ProcessingResolution;
		}
		if (IsDepthAnything3()) {
			return DepthAnything3ProcessingResolution;
		}
		if (IsPixelPerfectDepth()) {
			return PixelPerfectDepthProcessingResolution;
		}
		if (IsMoGe()) {
			return MoGeResolutionLevel;
		}
		return MurreProcessingResolution;
	}

	double MaxDepth() const {
		if (IsPriorDepthAnything()) {
			return PriorDAMaxDepth;
		}
		if (IsDepthAnythingV2()) {
			return DepthAnythingV2MaxDepth;
		}
		if (IsDepthAnything3()) {
			return DepthAnything3MaxDepth;
		}
		if (IsPixelPerfectDepth()) {
			return PixelPerfectDepthMaxDepth;
		}
		if (IsMoGe()) {
			return MoGeMaxDepth;
		}
		return MurreMaxDepth;
	}
};

struct DenseFusionConfig {
	bool Enable;
	std::string Device;
	double VoxelLength;
	double TruncVoxelMultiplier;
	double DepthMax;
	int ExtractEveryNFrames;
	std::string OutputMode;
	std::string ColorMode;
	int MinValidDepthPixels;
	size_t MaxDisplayPoints;
};

struct DepthAlignmentConfig {
	bool Enable;
	std::string Mode;
	int MinPoints;
	double MaxDepth;
	int Grid;
	double Overlap;
	bool LocalFit;
	std::string LocalModelMode;
	int MinBlockPoints;
	int MaxBlockSamples;
	bool SaveAlignDepth = true;
};

struct WeightedTSDFConfig {
	bool Enable;
	std::string GlobalP75Metric;
	int MinUpgradeSamples;
	double UnsupportedRegionScale;
	double HardRejectRatio;
	bool SoftSuppressionEnable;
	double SoftSuppressionExponent;
	double MaxVoxelWeight;
	double ExtractionWeightThreshold;
	double UpgradeCapDivisor;
	bool SaveQualityMaps;
};







struct GpuDiagnosticsConfig {
	bool Enable;
	int PollIntervalMs;
	bool UseNvidiaSmi;
	bool LogEventSnapshots;
	bool LogBundleAdjustmentStats;
};

struct FeedbackConfig {
	int StartCreateMesh;
	int Step;
};

struct TestConfig {
	std::string ModelFolder;
	int Start;
	int Step;
	int End;
};


class Config {
public:
	static Config& getInstance(const std::string& iniPath = "../settings.ini") {
		static std::once_flag flag;
		static std::unique_ptr<Config> instance;
		std::call_once(flag, [&]() {
			instance.reset(new Config(iniPath));
			});
		return *instance;
	}

	const std::string& getIniPath() const {
		return m_iniPath;
	}

	SfMConfig sfm;
	BundleAdjustmentRuntimeConfig bundleAdjustment;
	PipelineConfig pipeline;
	ExperimentConfig experiment;
	PythonConfig python;
	GlobalFeatureConfig globalFeature;
	DenseDepthConfig denseDepth;
	DepthAlignmentConfig depthAlignment;
	WeightedTSDFConfig weightedTSDF;
	DenseFusionConfig denseFusion;
	GpuDiagnosticsConfig gpuDiagnostics;
	FeedbackConfig feedback;
	TestConfig test;

	Config(const Config&) = delete;
	Config& operator=(const Config&) = delete;

private:
	std::string m_iniPath;

	Config(const std::string& iniPath) : m_iniPath(iniPath) {
		boost::property_tree::ptree pt;
		std::ifstream stream(iniPath, std::ios::binary);
		if (!stream.is_open()) {
			throw boost::property_tree::ini_parser::ini_parser_error(
				"cannot open file", iniPath, 0);
		}
		char bom[3] = {};
		stream.read(bom, 3);
		if (stream.gcount() != 3 ||
			static_cast<unsigned char>(bom[0]) != 0xEF ||
			static_cast<unsigned char>(bom[1]) != 0xBB ||
			static_cast<unsigned char>(bom[2]) != 0xBF) {
			stream.clear();
			stream.seekg(0);
		}
		boost::property_tree::ini_parser::read_ini(stream, pt);

		// [SfM]
		sfm.useFocalLength = pt.get<bool>("SfM.useFocalLength", false);
		sfm.useFactor = pt.get<double>("SfM.useFactor", 0.704);
		sfm.cameraModel = pt.get<std::string>("SfM.cameraModel", "SIMPLE_RADIAL");
		sfm.numLocalBAIters = std::max(1, pt.get<int>("SfM.numLocalBAIters", 2));
		sfm.maxNumFeatures = std::max<size_t>(1, pt.get<size_t>("SfM.maxNumFeatures", 4096));
		sfm.maxPendingImages = pt.get<size_t>("SfM.MaxPendingImages", 200);
		sfm.RegistrationRetryInterval = pt.get<int>("SfM.RegistrationRetryInterval", 50);
		sfm.RegistrationRetryIdleSeconds = pt.get<int>("SfM.RegistrationRetryIdleSeconds", 60);
		if (sfm.RegistrationRetryInterval <= 0 || sfm.RegistrationRetryIdleSeconds <= 0) {
			throw std::runtime_error("SfM registration retry interval and idle seconds must be positive");
		}

		// [BundleAdjustment]
		bundleAdjustment.LocalBACudaMinObservations = std::max(0, pt.get<int>("BundleAdjustment.LocalBACudaMinObservations", 80000));

		// [Pipeline]
		pipeline.AutoRunImages = pt.get<bool>("Pipeline.AutoRunImages", true);
		pipeline.AutoExportModel = pt.get<bool>("Pipeline.AutoExportModel", false);
		pipeline.InitialImagesNum = pt.get<int>("Pipeline.InitialImagesNum", 20);
		pipeline.Export3DGS = pt.get<bool>("Pipeline.Export3DGS", false);
		pipeline.OutputTimingFiles = pt.get<bool>("Pipeline.OutputTimingFiles", true);

		// [Experiment]
		experiment.Enable = pt.get<bool>("Experiment.Enable", false);
		experiment.ProjectPath = pt.get<std::string>("Experiment.ProjectPath", "");
		experiment.ImagePath = pt.get<std::string>("Experiment.ImagePath", "");
		experiment.ShowUI = pt.get<bool>("Experiment.ShowUI", true);
		experiment.AutoSaveIdleSeconds = std::max(1, pt.get<int>("Experiment.AutoSaveIdleSeconds", 60));

		// [Python]
		python.PythonHome = pt.get<std::string>("Python.PythonHome", "../../thirdparty/Python");
		python.PythonWorkspace = pt.get<std::string>("Python.PythonWorkspace", "../../scripts/pyscripts");

		// [GlobalFeature]
		globalFeature.GlobalFeatureType = pt.get<std::string>("GlobalFeature.GlobalFeatureType", "SupScene");
		// SupScene 配置
		globalFeature.SupSceneDevice = pt.get<std::string>("GlobalFeature.SupSceneDevice", "cuda");
		globalFeature.SupSceneWeightsPath = pt.get<std::string>("GlobalFeature.SupSceneWeightsPath", "auto");

		// [DenseDepth]
		denseDepth.Enable = pt.get<bool>("DenseDepth.Enable", false);
		denseDepth.StartRegisteredImages = pt.get<int>("DenseDepth.StartRegisteredImages", 30);
		denseDepth.Method = pt.get<std::string>("DenseDepth.Method", "Murre");
		denseDepth.Device = pt.get<std::string>("DenseDepth.Device", "cuda");
		denseDepth.TimeoutMs = pt.get<int>("DenseDepth.TimeoutMs", 600000);
		denseDepth.ExportSfMBin = pt.get<bool>("DenseDepth.ExportSfMBin", false);
		denseDepth.OutputResolution = pt.get<int>("DenseDepth.OutputResolution", 784);
		// Murre配置
		denseDepth.MurreProcessingResolution = pt.get<int>("DenseDepth.MurreProcessingResolution", 784);
		denseDepth.MurreMaxDepth = pt.get<double>("DenseDepth.MurreMaxDepth", 50.0);
		denseDepth.MurreDenoiseSteps = pt.get<int>("DenseDepth.MurreDenoiseSteps", 4);
		denseDepth.MurreEnsembleSize = pt.get<int>("DenseDepth.MurreEnsembleSize", 1);

		// PriorDepthAnything配置
		denseDepth.PriorDAProcessingResolution = pt.get<int>("DenseDepth.PriorDAProcessingResolution", 784);
		denseDepth.PriorDAMaxDepth = pt.get<double>("DenseDepth.PriorDAMaxDepth", 50.0);
		denseDepth.PriorDAVersion = pt.get<std::string>("DenseDepth.PriorDAVersion", "1.1");
		denseDepth.PriorDAFrozenModelSize = pt.get<std::string>("DenseDepth.PriorDAFrozenModelSize", "vitb");
		denseDepth.PriorDAConditionedModelSize = pt.get<std::string>("DenseDepth.PriorDAConditionedModelSize", "vitb");
		denseDepth.PriorDAPattern = pt.get<std::string>("DenseDepth.PriorDAPattern", "");

		// DepthAnythingV2配置
		denseDepth.DepthAnythingV2ProcessingResolution = pt.get<int>("DenseDepth.DepthAnythingV2ProcessingResolution", 784);
		denseDepth.DepthAnythingV2MaxDepth = pt.get<double>("DenseDepth.DepthAnythingV2MaxDepth", 50.0);
		denseDepth.DepthAnythingV2Encoder = pt.get<std::string>("DenseDepth.DepthAnythingV2Encoder", "vitb");

		// DepthAnything3配置
		denseDepth.DepthAnything3Model = pt.get<std::string>("DenseDepth.DepthAnything3Model", "DA3MONO-LARGE");
		denseDepth.DepthAnything3ProcessingResolution = pt.get<int>("DenseDepth.DepthAnything3ProcessingResolution", 784);
		denseDepth.DepthAnything3MaxDepth = pt.get<double>("DenseDepth.DepthAnything3MaxDepth", 50.0);
		denseDepth.DepthAnything3ProcessResMethod = pt.get<std::string>("DenseDepth.DepthAnything3ProcessResMethod", "upper_bound_resize");
		denseDepth.DepthAnything3ConfThreshold = pt.get<double>("DenseDepth.DepthAnything3ConfThreshold", 0.0);

		// PixelPerfectDepth配置
		denseDepth.PixelPerfectDepthSemanticsModel = pt.get<std::string>("DenseDepth.PixelPerfectDepthSemanticsModel", "DA2");
		denseDepth.PixelPerfectDepthProcessingResolution = pt.get<int>("DenseDepth.PixelPerfectDepthProcessingResolution", 784);
		denseDepth.PixelPerfectDepthSamplingSteps = pt.get<int>("DenseDepth.PixelPerfectDepthSamplingSteps", 4);
		denseDepth.PixelPerfectDepthMaxDepth = pt.get<double>("DenseDepth.PixelPerfectDepthMaxDepth", 50.0);
		denseDepth.PixelPerfectDepthFP16 = pt.get<bool>("DenseDepth.PixelPerfectDepthFP16", true);

		// MoGe配置
		denseDepth.MoGeModel = pt.get<std::string>("DenseDepth.MoGeModel", "moge-2-vitl-normal");
		denseDepth.MoGeResolutionLevel = pt.get<int>("DenseDepth.MoGeResolutionLevel", 7);
		denseDepth.MoGeNumTokens = pt.get<int>("DenseDepth.MoGeNumTokens", 0);
		denseDepth.MoGeUseFP16 = pt.get<bool>("DenseDepth.MoGeUseFP16", true);
		denseDepth.MoGeUseSfMFov = pt.get<bool>("DenseDepth.MoGeUseSfMFov", true);
		denseDepth.MoGeApplyMask = pt.get<bool>("DenseDepth.MoGeApplyMask", true);
		denseDepth.MoGeMaskThreshold = pt.get<double>("DenseDepth.MoGeMaskThreshold", 0.5);
		denseDepth.MoGeSaveNormal = pt.get<bool>("DenseDepth.MoGeSaveNormal", true);
		denseDepth.MoGeMaxDepth = pt.get<double>("DenseDepth.MoGeMaxDepth", 50.0);

		// [DepthAlignment]
		depthAlignment.Enable = pt.get<bool>("DepthAlignment.Enable", false);
		depthAlignment.SaveAlignDepth = pt.get<bool>("DepthAlignment.SaveAlignDepth", true);
		depthAlignment.Mode = pt.get<std::string>("DepthAlignment.Mode", "auto");
		depthAlignment.MinPoints = pt.get<int>("DepthAlignment.MinPoints", 15);
		depthAlignment.MaxDepth = pt.get<double>("DepthAlignment.MaxDepth", 50.0);
		depthAlignment.Grid = pt.get<int>("DepthAlignment.Grid", 3);
		depthAlignment.Overlap = pt.get<double>("DepthAlignment.Overlap", 0.3);
		depthAlignment.LocalFit = pt.get<bool>("DepthAlignment.LocalFit", true);
		depthAlignment.LocalModelMode = pt.get<std::string>("DepthAlignment.LocalModelMode", "global");
		depthAlignment.MinBlockPoints = pt.get<int>("DepthAlignment.MinBlockPoints", 10);
		depthAlignment.MaxBlockSamples = pt.get<int>("DepthAlignment.MaxBlockSamples", 300);

		// [WeightedTSDF]
		weightedTSDF.Enable = pt.get<bool>("WeightedTSDF.Enable", true);
		weightedTSDF.GlobalP75Metric = pt.get<std::string>(
			"WeightedTSDF.GlobalP75Metric", "MSE");
		const auto metricFirst = weightedTSDF.GlobalP75Metric.find_first_not_of(
			" \t\r\n");
		const auto metricLast = weightedTSDF.GlobalP75Metric.find_last_not_of(
			" \t\r\n");
		if (metricFirst == std::string::npos || metricLast == std::string::npos) {
			throw std::invalid_argument(
				"WeightedTSDF.GlobalP75Metric must be E, RMSE, or MSE.");
		}
		weightedTSDF.GlobalP75Metric = weightedTSDF.GlobalP75Metric.substr(
			metricFirst, metricLast - metricFirst + 1);
		std::transform(
			weightedTSDF.GlobalP75Metric.begin(),
			weightedTSDF.GlobalP75Metric.end(),
			weightedTSDF.GlobalP75Metric.begin(),
			[](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
		if (weightedTSDF.GlobalP75Metric != "E" &&
			weightedTSDF.GlobalP75Metric != "RMSE" &&
			weightedTSDF.GlobalP75Metric != "MSE") {
			throw std::invalid_argument(
				"WeightedTSDF.GlobalP75Metric must be E, RMSE, or MSE.");
		}
		weightedTSDF.MinUpgradeSamples = pt.get<int>(
			"WeightedTSDF.MinUpgradeSamples", 20);
		weightedTSDF.UnsupportedRegionScale = pt.get<double>("WeightedTSDF.UnsupportedRegionScale", 0.10);
		weightedTSDF.HardRejectRatio = pt.get<double>("WeightedTSDF.HardRejectRatio", 0.25);
		weightedTSDF.SoftSuppressionEnable = pt.get<bool>(
			"WeightedTSDF.SoftSuppressionEnable", true);
		weightedTSDF.SoftSuppressionExponent = pt.get<double>("WeightedTSDF.SoftSuppressionExponent", 2.0);
		weightedTSDF.MaxVoxelWeight = pt.get<double>("WeightedTSDF.MaxVoxelWeight", 0.0);
		weightedTSDF.ExtractionWeightThreshold = pt.get<double>(
			"WeightedTSDF.ExtractionWeightThreshold", 3.0);
		weightedTSDF.UpgradeCapDivisor = pt.get<double>(
			"WeightedTSDF.UpgradeCapDivisor", 3.0);
		weightedTSDF.SaveQualityMaps = pt.get<bool>("WeightedTSDF.SaveQualityMaps", true);
		if (weightedTSDF.ExtractionWeightThreshold <= 0.0) {
			throw std::invalid_argument(
				"WeightedTSDF.ExtractionWeightThreshold must be greater than 0 "
				"for Global P75 AlignGrid.");
		}
		if (!std::isfinite(weightedTSDF.UpgradeCapDivisor) ||
			weightedTSDF.UpgradeCapDivisor < 1.0) {
			throw std::invalid_argument(
				"WeightedTSDF.UpgradeCapDivisor must be finite and at least 1.");
		}
		if (weightedTSDF.MinUpgradeSamples < 0)
			throw std::invalid_argument("WeightedTSDF.MinUpgradeSamples must be non-negative.");
		if (!std::isfinite(weightedTSDF.UnsupportedRegionScale) ||
			weightedTSDF.UnsupportedRegionScale < 0.0 || weightedTSDF.UnsupportedRegionScale > 1.0)
			throw std::invalid_argument("WeightedTSDF.UnsupportedRegionScale must be finite and in [0, 1].");
		if (!std::isfinite(weightedTSDF.HardRejectRatio) ||
			weightedTSDF.HardRejectRatio < 0.0 || weightedTSDF.HardRejectRatio > 1.0)
			throw std::invalid_argument("WeightedTSDF.HardRejectRatio must be finite and in [0, 1].");
		if (!std::isfinite(weightedTSDF.SoftSuppressionExponent) || weightedTSDF.SoftSuppressionExponent <= 0.0)
			throw std::invalid_argument("WeightedTSDF.SoftSuppressionExponent must be finite and greater than 0.");
		if (!std::isfinite(weightedTSDF.MaxVoxelWeight) || weightedTSDF.MaxVoxelWeight < 0.0)
			throw std::invalid_argument("WeightedTSDF.MaxVoxelWeight must be finite and non-negative.");
		if (!std::isfinite(weightedTSDF.ExtractionWeightThreshold) || weightedTSDF.ExtractionWeightThreshold < 0.0)
			throw std::invalid_argument("WeightedTSDF.ExtractionWeightThreshold must be finite and non-negative.");

		// [DenseFusion]
		denseFusion.Enable = pt.get<bool>("DenseFusion.Enable", true);
		denseFusion.Device = pt.get<std::string>("DenseFusion.Device", "CPU");
		denseFusion.VoxelLength = pt.get<double>("DenseFusion.VoxelLength", 0.04);
		denseFusion.TruncVoxelMultiplier = pt.get<double>("DenseFusion.TruncVoxelMultiplier", 8.0);
		denseFusion.DepthMax = pt.get<double>("DenseFusion.DepthMax", 50.0);
		denseFusion.ExtractEveryNFrames = pt.get<int>("DenseFusion.ExtractEveryNFrames", 1);
		denseFusion.OutputMode = pt.get<std::string>("DenseFusion.OutputMode", "PointCloud");
		denseFusion.ColorMode = pt.get<std::string>("DenseFusion.ColorMode", "White");
		denseFusion.MinValidDepthPixels = pt.get<int>("DenseFusion.MinValidDepthPixels", 50);
		denseFusion.MaxDisplayPoints = pt.get<size_t>("DenseFusion.MaxDisplayPoints", 100000000);

		// [Feedback]
		feedback.StartCreateMesh = pt.get<int>("Feedback.StartCreateMesh", 4000);
		feedback.Step = pt.get<int>("Feedback.Step", 10);

		// [GpuDiagnostics]
		gpuDiagnostics.Enable = pt.get<bool>("GpuDiagnostics.Enable", false);
		gpuDiagnostics.PollIntervalMs = pt.get<int>("GpuDiagnostics.PollIntervalMs", 2000);
		gpuDiagnostics.UseNvidiaSmi = pt.get<bool>("GpuDiagnostics.UseNvidiaSmi", true);
		gpuDiagnostics.LogEventSnapshots = pt.get<bool>("GpuDiagnostics.LogEventSnapshots", true);
		gpuDiagnostics.LogBundleAdjustmentStats = pt.get<bool>("GpuDiagnostics.LogBundleAdjustmentStats", true);

		// [Test]
		test.ModelFolder = pt.get<std::string>("Test.ModelFolder", "");
		test.Start = pt.get<int>("Test.Start", 80);
		test.Step = pt.get<int>("Test.Step", 10);
		test.End = pt.get<int>("Test.End", 110);
	}
};
