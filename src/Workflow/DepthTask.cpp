#include "DepthTask.h"

#include "../Base/config.h"
#include "../Base/GpuDiagnostics.h"
#include "../Base/misc.h"
#include "../Base/timer.h"
#include "../Scene/Camera.h"
#include "../Scene/Database.h"
#include "../Scene/Image.h"
#include "../Scene/Point2D.h"
#include "../Scene/Point3D.h"
#include "../Scene/ReconstructionManager.h"
#include "WorkData.h"

#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QProcess>
#include <QProcessEnvironment>
#include <QReadLocker>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kMinSparseDepth = 1e-3f;

struct DepthPredictionResult {
	bool ok = false;
	QString depthPath;
	QString colorPath;
	QString errorMessage;
};

struct DepthWorkerInitializeResult {
	bool ok = false;
	QString message;
	QString errorMessage;
};

std::string ToLower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

struct DenseDepthMethodPaths {
	std::string method;
	std::string displayName;
	std::string workerScript;
	std::string checkpointPath;
};

struct DenseDepthMethodOptions {
	int processingResolution = 768;
	double maxDepth = 50.0;
	int denoiseSteps = 4;
	int ensembleSize = 1;
	std::string version;
	std::string frozenModelSize;
	std::string conditionedModelSize;
	std::string pattern;
	std::string encoder;
};

struct NpyFloatImage {
	int width = 0;
	int height = 0;
	std::vector<float> data;
};

enum class SparseDepthMode {
	PriorDepthAnything,
	Murre
};

struct DepthFrameInfo {
	bool available = false;
	std::string sparseDepthPath;
	std::string imageFileName;
	std::string imageStem;
	int cameraWidth = 0;
	int cameraHeight = 0;
	int depthWidth = 0;
	int depthHeight = 0;
	Eigen::Matrix3d intrinsic = Eigen::Matrix3d::Identity();
	Eigen::Matrix4d camFromWorld = Eigen::Matrix4d::Identity();
	int numObservations = 0;
	int numValidIndices = 0;
	int numValidSparsePoints = 0;
};

DenseDepthMethodPaths ResolveDenseDepthMethod()
{
	const auto& config = Config::getInstance();
	const std::string method = config.denseDepth.Method.empty()
		? "Murre"
		: config.denseDepth.Method;
	const std::string methodLower = ToLower(method);
	const std::string workspace = StringReplace(
		EnsureTrailingSlash(config.python.PythonWorkspace),
		"\\",
		"/");
	const std::string root = workspace + "DepthEstimation/";

	if (methodLower == "murre") {
		return {
			"Murre",
			"Murre",
			root + "Murre/murre_depth.py",
			root + "Murre/checkpoints/Murre"
		};
	}

	if (methodLower == "priordepthanything" ||
		methodLower == "priorda" ||
		methodLower == "prior-depth-anything") {
		return {
			"PriorDepthAnything",
			"PriorDepthAnything",
			root + "PriorDepthAnything/prior_depth_anything_worker.py",
			root + "PriorDepthAnything/checkpoints"
		};
	}

	if (methodLower == "depthanythingv2" ||
		methodLower == "depthanything-v2" ||
		methodLower == "depth-anything-v2" ||
		methodLower == "dav2") {
		return {
			"DepthAnythingV2",
			"DepthAnythingV2",
			root + "DepthAnythingV2/depth_anything_v2_worker.py",
			root + "DepthAnythingV2/checkpoints"
		};
	}

	if (methodLower == "depthanything3" ||
		methodLower == "depthanything-v3" ||
		methodLower == "depth-anything-3" ||
		methodLower == "da3") {
		return {
			"DepthAnything3",
			"DepthAnything3",
			root + "DepthAnything3/depth_anything3_worker.py",
			root + "DepthAnything3/checkpoints"
		};
	}

	if (methodLower == "pixelperfectdepth" ||
		methodLower == "pixel-perfect-depth" ||
		methodLower == "ppd") {
		return {
			"PixelPerfectDepth",
			"PixelPerfectDepth",
			root + "PixelPerfectDepth/pixel_perfect_depth_worker.py",
			root + "PixelPerfectDepth/checkpoints"
		};
	}

	if (methodLower == "moge" ||
		methodLower == "moge2" ||
		methodLower == "moge-2") {
		return {
			"MoGe",
			"MoGe",
			root + "MoGe/moge_worker.py",
			root + "MoGe/checkpoints"
		};
	}

	throw std::runtime_error("Unsupported dense depth method: " + method);
}

DenseDepthMethodOptions ResolveDenseDepthMethodOptions(const DenseDepthMethodPaths& methodPaths)
{
	const auto& depth = Config::getInstance().denseDepth;
	const std::string methodLower = ToLower(methodPaths.method);

	if (methodLower == "priordepthanything") {
		return {
			depth.PriorDAProcessingResolution,
			depth.PriorDAMaxDepth,
			0,
			0,
			depth.PriorDAVersion,
			depth.PriorDAFrozenModelSize,
			depth.PriorDAConditionedModelSize,
			depth.PriorDAPattern,
			""
		};
	}

	if (methodLower == "depthanythingv2") {
		return {
			depth.DepthAnythingV2ProcessingResolution,
			depth.DepthAnythingV2MaxDepth,
			0,
			0,
			"",
			"",
			"",
			"",
			depth.DepthAnythingV2Encoder
		};
	}

	if (methodLower == "depthanything3") {
		return {
			depth.DepthAnything3ProcessingResolution,
			depth.DepthAnything3MaxDepth,
			0,
			0,
			"",
			"",
			"",
			"",
			""
		};
	}

	if (methodLower == "pixelperfectdepth") {
		return {
			depth.PixelPerfectDepthProcessingResolution,
			depth.PixelPerfectDepthMaxDepth,
			depth.PixelPerfectDepthSamplingSteps,
			0,
			"",
			"",
			"",
			"",
			""
		};
	}

	if (methodLower == "moge") {
		return {
			depth.MoGeResolutionLevel,
			depth.MoGeMaxDepth,
			0,
			0,
			"",
			"",
			"",
			"",
			""
		};
	}

	return {
		depth.MurreProcessingResolution,
		depth.MurreMaxDepth,
		depth.MurreDenoiseSteps,
		depth.MurreEnsembleSize,
		"",
		"",
		"",
		"",
		""
	};
}

bool IsPriorDepthMethod(const DenseDepthMethodPaths& methodPaths)
{
	return ToLower(methodPaths.method) == "priordepthanything";
}

bool IsMurreDepthMethod(const DenseDepthMethodPaths& methodPaths)
{
	return ToLower(methodPaths.method) == "murre";
}

bool NeedsSparseDepthPrior(const DenseDepthMethodPaths& methodPaths)
{
	return IsPriorDepthMethod(methodPaths) || IsMurreDepthMethod(methodPaths);
}

void WriteLittleEndianU16(std::ofstream& stream, const uint16_t value)
{
	const char bytes[2] = {
		static_cast<char>(value & 0xff),
		static_cast<char>((value >> 8) & 0xff)
	};
	stream.write(bytes, 2);
}

void WriteNpyFloatImage(const std::string& path, const NpyFloatImage& image)
{
	std::ofstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("Cannot write sparse depth npy: " + path);
	}

	std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
		std::to_string(image.height) + ", " + std::to_string(image.width) + "), }";
	const size_t preambleSize = 10;
	const size_t padding = 16 - ((preambleSize + header.size() + 1) % 16);
	header.append(padding, ' ');
	header.push_back('\n');

	if (header.size() > std::numeric_limits<uint16_t>::max()) {
		throw std::runtime_error("Sparse depth npy header is too large.");
	}

	stream.write("\x93NUMPY", 6);
	stream.put(1);
	stream.put(0);
	WriteLittleEndianU16(stream, static_cast<uint16_t>(header.size()));
	stream.write(header.data(), static_cast<std::streamsize>(header.size()));
	stream.write(
		reinterpret_cast<const char*>(image.data.data()),
		static_cast<std::streamsize>(image.data.size() * sizeof(float)));
}

std::string StemFromImageName(const std::string& imageName)
{
	std::string root;
	std::string ext;
	SplitFileExtension(imageName, &root, &ext);
	return root;
}

QJsonArray Matrix3ToJson(const Eigen::Matrix3d& matrix)
{
	QJsonArray rows;
	for (int r = 0; r < 3; ++r) {
		QJsonArray row;
		for (int c = 0; c < 3; ++c) {
			row.append(matrix(r, c));
		}
		rows.append(row);
	}
	return rows;
}

QJsonArray Matrix4ToJson(const Eigen::Matrix4d& matrix)
{
	QJsonArray rows;
	for (int r = 0; r < 4; ++r) {
		QJsonArray row;
		for (int c = 0; c < 4; ++c) {
			row.append(matrix(r, c));
		}
		rows.append(row);
	}
	return rows;
}

DepthFrameInfo BuildDepthFrameInfo(
	const std::shared_ptr<Reconstruction>& model,
	const image_t imageID)
{
	if (!model || !model->ExistsImage(imageID) ||
		!model->IsImageRegistered(imageID)) {
		throw std::runtime_error("Image is not registered for dense depth frame info.");
	}

	const Image& image = model->Image(imageID);
	if (!model->ExistsCamera(image.CameraId())) {
		throw std::runtime_error("Camera is missing for dense depth frame info.");
	}
	const Camera& camera = model->Camera(image.CameraId());

	const int cameraWidth = static_cast<int>(camera.Width());
	const int cameraHeight = static_cast<int>(camera.Height());
	if (cameraWidth <= 0 || cameraHeight <= 0) {
		throw std::runtime_error("Camera has invalid dimensions for dense depth frame info.");
	}

	DepthFrameInfo info;
	info.available = true;
	info.imageFileName = image.Name();
	info.imageStem = StemFromImageName(image.Name());
	info.cameraWidth = cameraWidth;
	info.cameraHeight = cameraHeight;
	info.intrinsic = camera.CalibrationMatrix();
	info.camFromWorld.setIdentity();
	info.camFromWorld.topRows<3>() = image.CamFromWorld().ToMatrix();
	info.numObservations = static_cast<int>(image.Points2D().size());
	return info;
}

void WriteSparseDepth(
	const std::shared_ptr<Reconstruction>& model,
	const image_t imageID,
	const std::string& outputDir,
	const DenseDepthMethodOptions& options,
	const SparseDepthMode mode,
	DepthFrameInfo* info)
{
	if (!info || !info->available) {
		throw std::runtime_error("Depth frame info is missing for sparse depth prior.");
	}

	double scale = 1.0;
	int targetWidth = info->cameraWidth;
	int targetHeight = info->cameraHeight;
	if (options.processingResolution > 0) {
		scale = std::min(
			static_cast<double>(options.processingResolution) / info->cameraWidth,
			static_cast<double>(options.processingResolution) / info->cameraHeight);
		targetWidth = static_cast<int>(info->cameraWidth * scale);
		targetHeight = static_cast<int>(info->cameraHeight * scale);
	}

	const int divisor = mode == SparseDepthMode::Murre ? 16 : 14;
	const int depthWidth = targetWidth - targetWidth % divisor;
	const int depthHeight = targetHeight - targetHeight % divisor;
	if (depthWidth <= 0 || depthHeight <= 0) {
		throw std::runtime_error("Sparse depth prior has invalid output dimensions.");
	}

	NpyFloatImage sparseDepth;
	sparseDepth.width = depthWidth;
	sparseDepth.height = depthHeight;
	sparseDepth.data.assign(
		static_cast<size_t>(depthWidth) * depthHeight,
		0.0f);

	info->depthWidth = depthWidth;
	info->depthHeight = depthHeight;
	info->numValidIndices = 0;
	info->numValidSparsePoints = 0;

	Eigen::Matrix3d scaledIntrinsic = info->intrinsic;
	scaledIntrinsic.topRows<2>() *= scale;

	const Image& image = model->Image(imageID);

	for (const Point2D& point2D : image.Points2D()) {
		if (!point2D.HasPoint3D() ||
			!model->ExistsPoint3D(point2D.point3D_id)) {
			continue;
		}
		++info->numValidIndices;

		const Point3D& point3D = model->Point3D(point2D.point3D_id);
		if (mode == SparseDepthMode::Murre) {
			if ((point3D.HasError() && point3D.Error() >= 10.0) ||
				point3D.Track().Length() <= 1) {
				continue;
			}
		}

		const Eigen::Vector3d xyzCam = image.CamFromWorld() * point3D.XYZ();
		double z = xyzCam.z();
		if (!std::isfinite(z) || z <= kMinSparseDepth) {
			continue;
		}
		if (options.maxDepth > 0.0) {
			z = std::min(z, options.maxDepth);
		}

		int x = 0;
		int y = 0;
		if (mode == SparseDepthMode::Murre) {
			const Eigen::Vector3d pix = scaledIntrinsic * xyzCam;
			if (std::abs(pix.z()) <= std::numeric_limits<double>::epsilon()) {
				continue;
			}
			x = static_cast<int>(pix.x() / pix.z());
			y = static_cast<int>(pix.y() / pix.z());
		}
		else {
			x = static_cast<int>(std::lround(point2D.xy.x() * scale));
			y = static_cast<int>(std::lround(point2D.xy.y() * scale));
		}
		if (x < 0 || x >= depthWidth || y < 0 || y >= depthHeight) {
			continue;
		}

		float& value = sparseDepth.data[static_cast<size_t>(y) * depthWidth + x];
		if (value <= 0.0f || z < value) {
			value = static_cast<float>(z);
		}
	}

	for (const float value : sparseDepth.data) {
		if (value > 0.0f) {
			++info->numValidSparsePoints;
		}
	}

	const std::string sparseDepthDir = outputDir + "/sparse_depth";
	CreateDirIfNotExists(sparseDepthDir, true);
	info->sparseDepthPath = sparseDepthDir + "/" + info->imageStem + "_sparse_depth.npy";
	WriteNpyFloatImage(info->sparseDepthPath, sparseDepth);
}

class DenseDepthWorkerClient {
public:
	~DenseDepthWorkerClient()
	{
		Stop();
	}

	DepthWorkerInitializeResult Warmup()
	{
		std::lock_guard<std::mutex> lock(mutex_);

		DepthWorkerInitializeResult result;
		if (!EnsureStarted(&result.errorMessage)) {
			return result;
		}

		result.ok = true;
		result.message = QString("Depth estimator import succeeded. Depth estimation model: %1")
			.arg(QString::fromStdString(methodPaths_.displayName));
		return result;
	}

	DepthPredictionResult Predict(
		const image_t imageID,
		const std::string& sfmDir,
		const std::string& imageFolder,
		const std::string& outputDir,
		const DepthFrameInfo& frameInfo)
	{
		std::lock_guard<std::mutex> lock(mutex_);

		DepthPredictionResult result;
		if (!EnsureStarted(&result.errorMessage)) {
			return result;
		}

		const int requestID = ++requestCounter_;
		QJsonObject request;
		request["type"] = "predict";
		request["request_id"] = requestID;
		request["image_id"] = static_cast<int>(imageID);
		request["sfm_dir"] = QString::fromStdString(sfmDir);
		request["image_folder"] = QString::fromStdString(imageFolder);
		request["output_dir"] = QString::fromStdString(outputDir);
		request["output_timing"] = Config::getInstance().pipeline.OutputTimingFiles;
		request["gpu_diagnostics"] =
			Config::getInstance().pipeline.OutputTimingFiles &&
			Config::getInstance().gpuDiagnostics.Enable;
		if (frameInfo.available) {
			request["image_file_name"] =
				QString::fromStdString(frameInfo.imageFileName);
			request["image_name"] =
				QString::fromStdString(frameInfo.imageStem);
			request["camera_width"] = frameInfo.cameraWidth;
			request["camera_height"] = frameInfo.cameraHeight;
			request["depth_width"] = frameInfo.depthWidth;
			request["depth_height"] = frameInfo.depthHeight;
			request["intrinsic"] = Matrix3ToJson(frameInfo.intrinsic);
			request["pose"] = Matrix4ToJson(frameInfo.camFromWorld);
			request["num_observations"] = frameInfo.numObservations;
			request["num_valid_indices"] = frameInfo.numValidIndices;
			request["num_valid_sparse_points"] = frameInfo.numValidSparsePoints;
			if (!frameInfo.sparseDepthPath.empty()) {
				request["sparse_depth_path"] =
					QString::fromStdString(frameInfo.sparseDepthPath);
			}
		}

		const QByteArray payload =
			QJsonDocument(request).toJson(QJsonDocument::Compact) + "\n";
		if (process_.write(payload) == -1 || !process_.waitForBytesWritten(5000)) {
			result.errorMessage = QString("Failed to send request to %1 worker: %2")
				.arg(QString::fromStdString(methodPaths_.displayName))
				.arg(process_.errorString());
			Restart();
			return result;
		}

		const int timeoutMs = Config::getInstance().denseDepth.TimeoutMs;
		QByteArray lineBuffer;
		Timer timer;
		timer.Start();

		while (timer.ElapsedMicroSeconds() / 1000 < static_cast<size_t>(timeoutMs)) {
			if (process_.state() != QProcess::Running) {
				result.errorMessage =
					QString("%1 worker exited unexpectedly: %2")
					.arg(QString::fromStdString(methodPaths_.displayName))
					.arg(ReadStandardError());
				Restart();
				return result;
			}

			if (!process_.waitForReadyRead(200)) {
				continue;
			}

			lineBuffer += process_.readAllStandardOutput();
			while (true) {
				const int newline = lineBuffer.indexOf('\n');
				if (newline < 0) {
					break;
				}

				const QByteArray line = lineBuffer.left(newline).trimmed();
				lineBuffer.remove(0, newline + 1);
				if (line.isEmpty() || !line.startsWith('{')) {
					continue;
				}

				QJsonParseError parseError;
				const QJsonDocument responseDoc = QJsonDocument::fromJson(line, &parseError);
				if (parseError.error != QJsonParseError::NoError || !responseDoc.isObject()) {
					LOG(WARNING) << "Ignoring invalid dense depth worker line: " << line.constData();
					continue;
				}

				const QJsonObject response = responseDoc.object();
				if (response["type"].toString() == "ready") {
					continue;
				}
				if (response["request_id"].toInt() != requestID) {
					continue;
				}

				result.ok = response["ok"].toBool(false);
				result.depthPath = response["depth_path"].toString();
				result.colorPath = response["color_path"].toString();
				result.errorMessage = response["error"].toString();
				return result;
			}
		}

		result.errorMessage = QString("%1 worker timed out after %2 ms")
			.arg(QString::fromStdString(methodPaths_.displayName))
			.arg(timeoutMs);
		Restart();
		return result;
	}

private:
	bool EnsureStarted(QString* errorMessage)
	{
		if (process_.state() == QProcess::Running && isReady_) {
			return true;
		}

		const auto& config = Config::getInstance();
		try {
			methodPaths_ = ResolveDenseDepthMethod();
		}
		catch (const std::exception& e) {
			*errorMessage = QString::fromStdString(e.what());
			return false;
		}

		const QFileInfo pythonExe(QString::fromStdString(
			EnsureTrailingSlash(config.python.PythonHome) + "python.exe"));
		const QString program = pythonExe.exists()
			? pythonExe.absoluteFilePath()
			: "python";

		QStringList arguments;
		const DenseDepthMethodOptions methodOptions =
			ResolveDenseDepthMethodOptions(methodPaths_);
		arguments
			<< "-u"
			<< QString::fromStdString(methodPaths_.workerScript)
			<< "--worker"
			<< "--checkpoint" << QString::fromStdString(methodPaths_.checkpointPath)
			<< "--device" << QString::fromStdString(config.denseDepth.Device)
			<< "--processing-res" << QString::number(methodOptions.processingResolution)
			<< "--output-res" << QString::number(config.denseDepth.OutputResolution)
			<< "--max-depth" << QString::number(methodOptions.maxDepth)
			<< "--denoise-steps" << QString::number(methodOptions.denoiseSteps)
			<< "--ensemble-size" << QString::number(methodOptions.ensembleSize);
		if (!methodOptions.version.empty()) {
			arguments << "--version" << QString::fromStdString(methodOptions.version);
		}
		if (!methodOptions.frozenModelSize.empty()) {
			arguments << "--frozen-model-size" << QString::fromStdString(methodOptions.frozenModelSize);
		}
		if (!methodOptions.conditionedModelSize.empty()) {
			arguments << "--conditioned-model-size" << QString::fromStdString(methodOptions.conditionedModelSize);
		}
		if (!methodOptions.pattern.empty()) {
			arguments << "--pattern" << QString::fromStdString(methodOptions.pattern);
		}
		if (!methodOptions.encoder.empty()) {
			arguments << "--encoder" << QString::fromStdString(methodOptions.encoder);
		}
		if (ToLower(methodPaths_.method) == "pixelperfectdepth") {
			arguments
				<< "--fp16" << (config.denseDepth.PixelPerfectDepthFP16 ? "true" : "false")
				<< "--semantics-model" << QString::fromStdString(config.denseDepth.PixelPerfectDepthSemanticsModel);
		}
		if (ToLower(methodPaths_.method) == "depthanything3") {
			arguments
				<< "--model" << QString::fromStdString(config.denseDepth.DepthAnything3Model)
				<< "--process-res-method" << QString::fromStdString(config.denseDepth.DepthAnything3ProcessResMethod)
				<< "--save-pointmap" << "false"
				<< "--conf-threshold" << QString::number(config.denseDepth.DepthAnything3ConfThreshold);
		}
		if (ToLower(methodPaths_.method) == "moge") {
			arguments
				<< "--model" << QString::fromStdString(config.denseDepth.MoGeModel)
				<< "--resolution-level" << QString::number(config.denseDepth.MoGeResolutionLevel)
				<< "--num-tokens" << QString::number(config.denseDepth.MoGeNumTokens)
				<< "--fp16" << (config.denseDepth.MoGeUseFP16 ? "true" : "false")
				<< "--use-sfm-fov" << (config.denseDepth.MoGeUseSfMFov ? "true" : "false")
				<< "--apply-mask" << (config.denseDepth.MoGeApplyMask ? "true" : "false")
				<< "--mask-threshold" << QString::number(config.denseDepth.MoGeMaskThreshold)
				<< "--save-pointmap" << "false"
				<< "--save-normal" << (config.denseDepth.MoGeSaveNormal ? "true" : "false");
		}

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		const QFileInfo workerScript(QString::fromStdString(methodPaths_.workerScript));
		QString pythonPath = workerScript.absolutePath();
		if (env.contains("PYTHONPATH") && !env.value("PYTHONPATH").isEmpty()) {
			pythonPath += ";" + env.value("PYTHONPATH");
		}
		env.insert("PYTHONPATH", pythonPath);

		process_.setProcessEnvironment(env);
		process_.setProgram(program);
		process_.setArguments(arguments);
		process_.setProcessChannelMode(QProcess::SeparateChannels);
		process_.start();

		if (!process_.waitForStarted(10000)) {
			*errorMessage = QString("Failed to start %1 worker: %2")
				.arg(QString::fromStdString(methodPaths_.displayName))
				.arg(process_.errorString());
			return false;
		}
		GpuDiagnostics::RecordProcess(
			"DenseDepth:" + methodPaths_.displayName,
			static_cast<std::uint64_t>(process_.processId()));

		return WaitForReady(errorMessage);
	}

	bool WaitForReady(QString* errorMessage)
	{
		const int timeoutMs = Config::getInstance().denseDepth.TimeoutMs;
		QByteArray lineBuffer;
		Timer timer;
		timer.Start();

		while (timer.ElapsedMicroSeconds() / 1000 < static_cast<size_t>(timeoutMs)) {
			if (process_.state() != QProcess::Running) {
				*errorMessage = QString("%1 worker exited during initialization: %2")
					.arg(QString::fromStdString(methodPaths_.displayName))
					.arg(ReadStandardError());
				Restart();
				return false;
			}

			if (!process_.waitForReadyRead(200)) {
				continue;
			}

			lineBuffer += process_.readAllStandardOutput();
			while (true) {
				const int newline = lineBuffer.indexOf('\n');
				if (newline < 0) {
					break;
				}

				const QByteArray line = lineBuffer.left(newline).trimmed();
				lineBuffer.remove(0, newline + 1);
				if (line.isEmpty() || !line.startsWith('{')) {
					continue;
				}

				QJsonParseError parseError;
				const QJsonDocument responseDoc = QJsonDocument::fromJson(line, &parseError);
				if (parseError.error != QJsonParseError::NoError || !responseDoc.isObject()) {
					LOG(WARNING) << "Ignoring invalid dense depth worker initialization line: " << line.constData();
					continue;
				}

				const QJsonObject response = responseDoc.object();
				if (response["type"].toString() != "ready") {
					continue;
				}

				if (response["ok"].toBool(false)) {
					isReady_ = true;
					return true;
				}

				*errorMessage = response["error"].toString(
					QString("%1 depth model initialization failed.")
					.arg(QString::fromStdString(methodPaths_.displayName)));
				LOG(ERROR) << "[DenseDepth] " << errorMessage->toStdString();
				Restart();
				return false;
			}
		}

		*errorMessage = QString("%1 worker initialization timed out after %2 ms")
			.arg(QString::fromStdString(methodPaths_.displayName))
			.arg(timeoutMs);
		Restart();
		return false;
	}

	QString ReadStandardError()
	{
		return QString::fromLocal8Bit(process_.readAllStandardError()).trimmed();
	}

	void Stop()
	{
		if (process_.state() == QProcess::NotRunning) {
			isReady_ = false;
			return;
		}

		QJsonObject request;
		request["type"] = "shutdown";
		process_.write(QJsonDocument(request).toJson(QJsonDocument::Compact) + "\n");
		process_.waitForBytesWritten(1000);
		process_.closeWriteChannel();
		if (!process_.waitForFinished(3000)) {
			process_.kill();
			process_.waitForFinished(3000);
		}
		isReady_ = false;
	}

	void Restart()
	{
		Stop();
	}

	QProcess process_;
	std::mutex mutex_;
	std::atomic_int requestCounter_{ 0 };
	DenseDepthMethodPaths methodPaths_{ "Murre", "Murre", "", "" };
	bool isReady_ = false;
};

DenseDepthWorkerClient& DepthWorker()
{
	thread_local DenseDepthWorkerClient client;
	return client;
}

std::string DenseBasePath()
{
	std::string path;
	if (!Database::exportPath.empty()) {
		path = StringReplace(
			EnsureTrailingSlash(GetParentDir(Database::exportPath)),
			"\\",
			"/") + "DenseReconstruction";
	}
	else {
		path = StringReplace(
			EnsureTrailingSlash(Config::getInstance().python.PythonWorkspace),
			"\\",
			"/") + "DenseReconstruction";
	}
	GpuDiagnostics::SetOutputRoot(path);
	return path;
}

} // namespace

void DepthTask::run()
{
	try {
		const auto& config = Config::getInstance();
		if (!config.denseDepth.Enable) {
			return;
		}

		if (mode == Mode::Warmup) {
			const DepthWorkerInitializeResult result = DepthWorker().Warmup();
			if (result.ok) {
				emit depthWorkerInitialized(result.message);
			}
			else {
				const QString errorMessage = result.errorMessage.isEmpty()
					? QString("Unknown dense depth model initialization error")
					: result.errorMessage;
				LOG(ERROR) << "[DenseDepth] " << errorMessage.toStdString();
				emit depthWorkerInitializeFailed(errorMessage);
			}
			return;
		}

		const std::string basePath = DenseBasePath();
		const std::string sfmDir = basePath + "/DepthInput/bin";
		const std::string outputDir = basePath + "/Depth";
		const std::string imageFolder = StringReplace(Database::imageDir, "\\", "/");
		const DenseDepthMethodPaths methodPaths = ResolveDenseDepthMethod();
		const DenseDepthMethodOptions methodOptions =
			ResolveDenseDepthMethodOptions(methodPaths);
		DepthFrameInfo frameInfo;

		CreateDirIfNotExists(outputDir, true);

		{
			QReadLocker locker(&(m_data->m_models_lock[0]));
			if (m_data->getModelManager()->Size() == 0) {
				const QString message = "No reconstruction is available for dense depth prediction.";
				LOG(WARNING) << "[DenseDepth] " << message.toStdString();
				emit depthPredictionFailed(static_cast<quint32>(imageID), message);
				return;
			}

			const std::shared_ptr<Reconstruction>& model = m_data->getModelManager()->Get(0);
			if (!model || !model->ExistsImage(imageID) || !model->IsImageRegistered(imageID)) {
				const QString message = QString("Image %1 is not registered in model 0.").arg(imageID);
				LOG(WARNING) << "[DenseDepth] " << message.toStdString();
				emit depthPredictionFailed(static_cast<quint32>(imageID), message);
				return;
			}

			frameInfo = BuildDepthFrameInfo(model, imageID);

			if (NeedsSparseDepthPrior(methodPaths)) {
				const SparseDepthMode sparseMode = IsMurreDepthMethod(methodPaths)
					? SparseDepthMode::Murre
					: SparseDepthMode::PriorDepthAnything;
				WriteSparseDepth(
					model,
					imageID,
					outputDir,
					methodOptions,
					sparseMode,
					&frameInfo);
				LOG(INFO) << "[DenseDepth] Sparse depth prior for image " << imageID
					<< ": " << frameInfo.sparseDepthPath
					<< ", method=" << methodPaths.displayName
					<< ", size=" << frameInfo.depthWidth << "x" << frameInfo.depthHeight
					<< ", valid sparse pixels=" << frameInfo.numValidSparsePoints;
			}

			if (config.denseDepth.ExportSfMBin) {
				CreateDirIfNotExists(sfmDir, true);
				model->WriteBinary(sfmDir);
			}
		}

		LOG(INFO) << "[DenseDepth] Predicting depth for image " << imageID;
		Timer depthTimer;
		depthTimer.Start();
		const DepthPredictionResult result =
			DepthWorker().Predict(imageID, sfmDir, imageFolder, outputDir, frameInfo);
		const double depthTotalMs =
			static_cast<double>(depthTimer.ElapsedMicroSeconds()) / 1000.0;

		if (result.ok) {
			if (result.depthPath.isEmpty() || result.colorPath.isEmpty()) {
				throw std::runtime_error("Dense depth worker returned an empty depth or color path.");
			}
			LOG(INFO) << "[DenseDepth] Image " << imageID
				<< " depth predicted: " << result.depthPath.toStdString();
			if (config.pipeline.OutputTimingFiles) {
				const std::string header =
					"timestamp,image_id,image_name,global_feature_ms,retrieval_ms,"
					"sift_extract_ms,matching_ms,pose_estimation_ms,triangulation_ms,"
					"local_ba_ms,sfm_total_ms,depth_total_ms,depth_alignment_ms,"
					"tsdf_integrate_ms,tsdf_extract_ms,frame_elapsed_ms";
				std::ostringstream row;
				row << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
					<< imageID << ","
					<< GpuDiagnostics::EscapeCsvValue(frameInfo.imageStem) << ","
					<< ",,,,,,,,"
					<< depthTotalMs << ",,,,";
				GpuDiagnostics::AppendTimingRow(
					"pipeline_frame_timing.csv", header, row.str());
			}
			emit depthPredictionFinished(
				static_cast<quint32>(imageID),
				result.depthPath,
				result.colorPath);
			return;
		}

		const QString errorMessage = result.errorMessage.isEmpty()
			? QString("Unknown dense depth worker error")
			: result.errorMessage;
		LOG(ERROR) << "[DenseDepth] Image " << imageID
			<< " failed: " << errorMessage.toStdString();
		emit depthPredictionFailed(static_cast<quint32>(imageID), errorMessage);
	}
	catch (const std::exception& e) {
		const QString errorMessage = QString::fromStdString(e.what());
		if (mode == Mode::Warmup) {
			LOG(ERROR) << "[DenseDepth] Initialization failed: " << e.what();
			emit depthWorkerInitializeFailed(errorMessage);
		}
		else {
			LOG(ERROR) << "[DenseDepth] Image " << imageID
				<< " failed: " << e.what();
			emit depthPredictionFailed(static_cast<quint32>(imageID), errorMessage);
		}
	}
	catch (...) {
		const QString errorMessage = "Unknown dense depth task error.";
		if (mode == Mode::Warmup) {
			LOG(ERROR) << "[DenseDepth] Initialization failed: "
				<< errorMessage.toStdString();
			emit depthWorkerInitializeFailed(errorMessage);
		}
		else {
			LOG(ERROR) << "[DenseDepth] Image " << imageID
				<< " failed: " << errorMessage.toStdString();
			emit depthPredictionFailed(static_cast<quint32>(imageID), errorMessage);
		}
	}
}
