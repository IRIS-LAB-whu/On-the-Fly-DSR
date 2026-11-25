#include "DepthAlignmentTask.h"

#include "../Base/GpuDiagnostics.h"
#include "../Base/config.h"
#include "../Base/timer.h"
#include "../Estimator/RANSAC.h"
#include "../Feature/Bitmap.h"
#include "../Scene/Camera.h"
#include "../Scene/Image.h"
#include "../Scene/Point2D.h"
#include "../Scene/Point3D.h"
#include "../Scene/ReconstructionManager.h"
#include "WorkData.h"

#include <QDir>
#include <QReadLocker>
#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kMinDepth = 1e-3f;

struct NpyFloatImage {
	int width = 0;
	int height = 0;
	std::vector<float> data;
};

struct SparseAlignmentSample {
	int x = 0;
	int y = 0;
	float rawDepth = 0.0f;
	float metricDepth = 0.0f;
	double trackLength = 0.0;
	double reprojectionError = 0.0;
};

enum class AlignmentMode {
	Linear,
	Inverse,
	Log,
	Quadratic,
	InvQuad
};

struct DepthModel {
	AlignmentMode mode = AlignmentMode::Linear;
	Eigen::Vector3d coeffs = Eigen::Vector3d::Zero();
	int degree = 1;
	bool valid = false;
};

std::string ToLower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

std::string ReadExact(std::ifstream& stream, const size_t size)
{
	std::string buffer(size, '\0');
	stream.read(buffer.data(), static_cast<std::streamsize>(size));
	if (stream.gcount() != static_cast<std::streamsize>(size)) {
		throw std::runtime_error("Unexpected end of npy file.");
	}
	return buffer;
}

uint16_t ReadLittleEndianU16(std::ifstream& stream)
{
	const std::string bytes = ReadExact(stream, 2);
	return static_cast<uint16_t>(
		static_cast<unsigned char>(bytes[0]) |
		(static_cast<unsigned char>(bytes[1]) << 8));
}

uint32_t ReadLittleEndianU32(std::ifstream& stream)
{
	const std::string bytes = ReadExact(stream, 4);
	return static_cast<uint32_t>(
		static_cast<unsigned char>(bytes[0]) |
		(static_cast<unsigned char>(bytes[1]) << 8) |
		(static_cast<unsigned char>(bytes[2]) << 16) |
		(static_cast<unsigned char>(bytes[3]) << 24));
}

std::vector<int> ParseShape(const std::string& header)
{
	const size_t begin = header.find('(');
	const size_t end = header.find(')', begin);
	if (begin == std::string::npos || end == std::string::npos || end <= begin) {
		throw std::runtime_error("Invalid npy shape header.");
	}

	std::vector<int> shape;
	std::stringstream ss(header.substr(begin + 1, end - begin - 1));
	while (ss.good()) {
		std::string item;
		std::getline(ss, item, ',');
		item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
		if (!item.empty()) {
			shape.push_back(std::stoi(item));
		}
	}
	return shape;
}

NpyFloatImage ReadNpyFloatImage(const std::string& path)
{
	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("Cannot open raw depth npy: " + path);
	}

	const std::string magic = ReadExact(stream, 6);
	if (magic != "\x93NUMPY") {
		throw std::runtime_error("Invalid npy magic: " + path);
	}

	const uint8_t major = static_cast<uint8_t>(ReadExact(stream, 1)[0]);
	ReadExact(stream, 1);

	uint32_t headerLength = 0;
	if (major == 1) {
		headerLength = ReadLittleEndianU16(stream);
	}
	else if (major == 2 || major == 3) {
		headerLength = ReadLittleEndianU32(stream);
	}
	else {
		throw std::runtime_error("Unsupported npy version.");
	}

	const std::string header = ReadExact(stream, headerLength);
	if (header.find("'descr': '<f4'") == std::string::npos &&
		header.find("\"descr\": \"<f4\"") == std::string::npos) {
		throw std::runtime_error("Only little-endian float32 npy depth is supported.");
	}
	if (header.find("True") != std::string::npos) {
		throw std::runtime_error("Fortran-order npy depth is not supported.");
	}

	const std::vector<int> shape = ParseShape(header);
	if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
		throw std::runtime_error("Depth npy must have shape HxW.");
	}

	NpyFloatImage image;
	image.height = shape[0];
	image.width = shape[1];
	image.data.resize(static_cast<size_t>(image.width) * image.height);
	stream.read(
		reinterpret_cast<char*>(image.data.data()),
		static_cast<std::streamsize>(image.data.size() * sizeof(float)));
	if (stream.gcount() != static_cast<std::streamsize>(image.data.size() * sizeof(float))) {
		throw std::runtime_error("Depth npy payload is incomplete.");
	}
	return image;
}

void WriteLittleEndianU16(std::ofstream& stream, const uint16_t value)
{
	const char bytes[2] = {
		static_cast<char>(value & 0xff),
		static_cast<char>((value >> 8) & 0xff)
	};
	stream.write(bytes, 2);
}

void WriteNpyFloatImage(
	const std::string& path,
	const int width,
	const int height,
	const std::vector<float>& data)
{
	std::ofstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("Cannot write depth npy: " + path);
	}

	std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
		std::to_string(height) + ", " + std::to_string(width) + "), }";
	const size_t preambleSize = 10;
	const size_t padding = 16 - ((preambleSize + header.size() + 1) % 16);
	header.append(padding, ' ');
	header.push_back('\n');

	if (header.size() > std::numeric_limits<uint16_t>::max()) {
		throw std::runtime_error("Npy header is too large.");
	}

	stream.write("\x93NUMPY", 6);
	stream.put(1);
	stream.put(0);
	WriteLittleEndianU16(stream, static_cast<uint16_t>(header.size()));
	stream.write(header.data(), static_cast<std::streamsize>(header.size()));
	stream.write(
		reinterpret_cast<const char*>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(float)));
	stream.close();
	if (!stream) {
		throw std::runtime_error("Failed to finish writing depth npy: " + path);
	}
}

void WriteNpyFloatImage(const std::string& path, const NpyFloatImage& image)
{
	WriteNpyFloatImage(path, image.width, image.height, image.data);
}

std::string ParentPath(const std::string& path)
{
	const size_t pos = path.find_last_of("/\\");
	if (pos == std::string::npos) {
		return "";
	}
	return path.substr(0, pos);
}

std::string FileName(const std::string& path)
{
	const size_t pos = path.find_last_of("/\\");
	if (pos == std::string::npos) {
		return path;
	}
	return path.substr(pos + 1);
}

std::string MakeAlignedDepthPath(const std::string& rawDepthPath)
{
	const std::string npDir = ParentPath(rawDepthPath);
	const std::string depthRoot = ParentPath(npDir);
	const char sep = depthRoot.find('\\') != std::string::npos ? '\\' : '/';
	const std::string outputDir = depthRoot + sep + "aligned_np";
	QDir().mkpath(QString::fromStdString(outputDir));
	return outputDir + sep + FileName(rawDepthPath);
}

void SaveAlignmentSparseDepth(
	const bool enabled,
	const image_t imageID,
	const std::string& rawDepthPath,
	const int width,
	const int height,
	const std::vector<float>& sparseMetric)
{
	if (!enabled) {
		return;
	}
	try {
		Timer timer;
		timer.Start();
		const std::string depthRoot = ParentPath(ParentPath(rawDepthPath));
		const std::string outputDir = depthRoot + "/alignment_sparse_depth";
		if (!QDir().mkpath(QString::fromStdString(outputDir))) {
			throw std::runtime_error("Cannot create alignment sparse depth directory: " + outputDir);
		}
		std::string stem = FileName(rawDepthPath);
		const std::string suffix = "_depth.npy";
		if (stem.size() >= suffix.size() &&
			stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
			stem.resize(stem.size() - suffix.size());
		}
		else if (stem.size() >= 4 && stem.compare(stem.size() - 4, 4, ".npy") == 0) {
			stem.resize(stem.size() - 4);
		}
		const std::string path = outputDir + "/" + stem + "_sparse_depth.npy";
		WriteNpyFloatImage(path, width, height, sparseMetric);
		const size_t validPixels = static_cast<size_t>(std::count_if(
			sparseMetric.begin(), sparseMetric.end(),
			[](const float value) { return std::isfinite(value) && value > kMinDepth; }));
		LOG(INFO) << "[DepthAlignment] Image " << imageID
			<< " alignment sparse depth saved: " << path
			<< ", size=" << width << "x" << height
			<< ", valid sparse pixels=" << validPixels
			<< ", write_ms=" << static_cast<double>(timer.ElapsedMicroSeconds()) / 1000.0;
	}
	catch (const std::exception& error) {
		LOG(WARNING) << "[DepthAlignment] Image " << imageID
			<< " could not save alignment sparse depth: " << error.what()
			<< "; continuing depth alignment.";
	}
}

std::pair<std::string, std::string> MakeAlignmentMapPaths(
	const std::string& rawDepthPath,
	const std::string& mapSuffix)
{
	const std::string npDir = ParentPath(rawDepthPath);
	const std::string depthRoot = ParentPath(npDir);
	const char sep = depthRoot.find('\\') != std::string::npos ? '\\' : '/';
	const std::string outputDir = depthRoot + sep + "alignment_weight";
	QDir().mkpath(QString::fromStdString(outputDir));
	std::string baseName = FileName(rawDepthPath);
	const std::string suffix = "_depth.npy";
	if (baseName.size() >= suffix.size() &&
		baseName.compare(baseName.size() - suffix.size(), suffix.size(), suffix) == 0) {
		baseName.resize(baseName.size() - suffix.size());
	}
	return {
		outputDir + sep + baseName + mapSuffix + ".npy",
		outputDir + sep + baseName + mapSuffix + ".png"
	};
}

void SaveAlignmentFloatMap(
	const std::vector<float>& values,
	const int width,
	const int height,
	const std::string& rawDepthPath,
	const std::string& mapSuffix,
	const bool normalizeForPng)
{
	NpyFloatImage image;
	image.width = width;
	image.height = height;
	image.data = values;
	const auto paths = MakeAlignmentMapPaths(rawDepthPath, mapSuffix);
	WriteNpyFloatImage(paths.first, image);

	Bitmap bitmap;
	if (!bitmap.Allocate(width, height, false)) {
		throw std::runtime_error("Cannot allocate alignment quality PNG.");
	}
	float displayMax = 1.0f;
	if (normalizeForPng && !values.empty()) {
		displayMax = std::max(
			1e-6f, *std::max_element(values.begin(), values.end()));
	}
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const float q = std::clamp(
				values[static_cast<size_t>(y) * width + x] / displayMax,
				0.0f, 1.0f);
			bitmap.SetPixel(
				x,
				y,
				BitmapColor<uint8_t>(static_cast<uint8_t>(std::lround(q * 255.0f))));
		}
	}
	if (!bitmap.Write(paths.second)) {
		throw std::runtime_error("Cannot write alignment quality PNG: " + paths.second);
	}
}

bool ParseMode(const std::string& value, AlignmentMode* mode)
{
	const std::string lower = ToLower(value);
	if (lower == "linear") {
		*mode = AlignmentMode::Linear;
		return true;
	}
	if (lower == "inverse") {
		*mode = AlignmentMode::Inverse;
		return true;
	}
	if (lower == "log") {
		*mode = AlignmentMode::Log;
		return true;
	}
	if (lower == "quadratic") {
		*mode = AlignmentMode::Quadratic;
		return true;
	}
	if (lower == "inv_quad" || lower == "invquad") {
		*mode = AlignmentMode::InvQuad;
		return true;
	}
	return false;
}

std::string ModeName(const AlignmentMode mode)
{
	switch (mode) {
	case AlignmentMode::Linear:
		return "linear";
	case AlignmentMode::Inverse:
		return "inverse";
	case AlignmentMode::Log:
		return "log";
	case AlignmentMode::Quadratic:
		return "quadratic";
	case AlignmentMode::InvQuad:
		return "inv_quad";
	default:
		return "unknown";
	}
}

int ModelDegree(const AlignmentMode mode)
{
	return mode == AlignmentMode::Quadratic || mode == AlignmentMode::InvQuad ? 2 : 1;
}

double FeatureValue(const double rawDepth, const AlignmentMode mode, const int column)
{
	if (ModelDegree(mode) == 2) {
		if (column == 0) {
			return rawDepth * rawDepth;
		}
		if (column == 1) {
			return rawDepth;
		}
		return 1.0;
	}

	if (column == 0) {
		return mode == AlignmentMode::Log ? std::log(rawDepth) : rawDepth;
	}
	return 1.0;
}

bool TransformTarget(const double metricDepth, const AlignmentMode mode, double* target)
{
	if (!std::isfinite(metricDepth) || metricDepth <= kMinDepth) {
		return false;
	}
	if (mode == AlignmentMode::Inverse || mode == AlignmentMode::InvQuad) {
		*target = 1.0 / metricDepth;
		return std::isfinite(*target);
	}
	if (mode == AlignmentMode::Log) {
		*target = std::log(metricDepth);
		return std::isfinite(*target);
	}
	*target = metricDepth;
	return true;
}

double PredictDepth(const double rawDepth, const DepthModel& model)
{
	if (!model.valid || !std::isfinite(rawDepth) || rawDepth <= kMinDepth) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	double transformed = 0.0;
	for (int col = 0; col <= model.degree; ++col) {
		transformed += model.coeffs[col] * FeatureValue(rawDepth, model.mode, col);
	}

	if (model.mode == AlignmentMode::Inverse || model.mode == AlignmentMode::InvQuad) {
		if (std::abs(transformed) < 1e-12) {
			return std::numeric_limits<double>::quiet_NaN();
		}
		return 1.0 / transformed;
	}
	if (model.mode == AlignmentMode::Log) {
		if (transformed > 50.0 || transformed < -50.0) {
			return std::numeric_limits<double>::quiet_NaN();
		}
		return std::exp(transformed);
	}
	return transformed;
}

class DepthAlignmentEstimator {
public:
	using X_t = double;
	using Y_t = double;
	using M_t = DepthModel;
	static constexpr int kMinNumSamples = 6;

	std::vector<M_t> Estimate(
		const std::vector<X_t>& rawDepths,
		const std::vector<Y_t>& metricDepths) const
	{
		if (rawDepths.size() != metricDepths.size() ||
			rawDepths.size() < static_cast<size_t>(kMinNumSamples)) {
			return {};
		}

		const int degree = ModelDegree(mode);
		const int cols = degree + 1;
		Eigen::MatrixXd A(rawDepths.size(), cols);
		Eigen::VectorXd b(rawDepths.size());

		for (size_t row = 0; row < rawDepths.size(); ++row) {
			if (!std::isfinite(rawDepths[row]) || rawDepths[row] <= kMinDepth) {
				return {};
			}
			for (int col = 0; col < cols; ++col) {
				A(static_cast<Eigen::Index>(row), col) =
					FeatureValue(rawDepths[row], mode, col);
			}
			double target = 0.0;
			if (!TransformTarget(metricDepths[row], mode, &target)) {
				return {};
			}
			b(static_cast<Eigen::Index>(row)) = target;
		}

		const Eigen::VectorXd solution = A.colPivHouseholderQr().solve(b);
		if (solution.size() != cols || !solution.allFinite()) {
			return {};
		}

		M_t model;
		model.mode = mode;
		model.degree = degree;
		model.valid = true;
		model.coeffs.setZero();
		for (int i = 0; i < cols; ++i) {
			model.coeffs[i] = solution[i];
		}
		return { model };
	}

	void Residuals(
		const std::vector<X_t>& rawDepths,
		const std::vector<Y_t>& metricDepths,
		const M_t& model,
		std::vector<double>* residuals) const
	{
		residuals->resize(rawDepths.size());
		for (size_t i = 0; i < rawDepths.size(); ++i) {
			double target = 0.0;
			if (!model.valid ||
				!std::isfinite(rawDepths[i]) ||
				rawDepths[i] <= kMinDepth ||
				!TransformTarget(metricDepths[i], model.mode, &target)) {
				(*residuals)[i] = std::numeric_limits<double>::max();
				continue;
			}

			double transformed = 0.0;
			for (int col = 0; col <= model.degree; ++col) {
				transformed += model.coeffs[col] *
					FeatureValue(rawDepths[i], model.mode, col);
			}
			const double residual = target - transformed;
			(*residuals)[i] = residual * residual;
		}
	}

	AlignmentMode mode = AlignmentMode::Linear;
};

double RansacThreshold(const AlignmentMode mode)
{
	if (mode == AlignmentMode::Inverse || mode == AlignmentMode::InvQuad) {
		return 0.1;
	}
	if (mode == AlignmentMode::Log) {
		return 0.2;
	}
	return 0.5;
}

bool FitMode(
	const AlignmentMode mode,
	const std::vector<double>& rawDepths,
	const std::vector<double>& metricDepths,
	const int minPoints,
	DepthModel* bestModel,
	double* bestRmse,
	double* bestMeanResidual,
	double* bestRelativeVariance,
	size_t* bestInliers)
{
	RANSACOptions options;
	options.max_error = RansacThreshold(mode);
	options.min_inlier_ratio = 0.1;
	options.confidence = 0.999;
	options.dyn_num_trials_multiplier = 3.0;
	options.min_num_trials = 100;
	options.max_num_trials = 10000;

	RANSAC<DepthAlignmentEstimator> ransac(options);
	ransac.estimator.mode = mode;
	const auto report = ransac.Estimate(rawDepths, metricDepths);
	if (!report.success ||
		report.support.num_inliers < static_cast<size_t>(std::max(DepthAlignmentEstimator::kMinNumSamples, minPoints / 2))) {
		return false;
	}

	double squaredSum = 0.0;
	double residualSum = 0.0;
	double relativeSum = 0.0;
	double relativeSquaredSum = 0.0;
	size_t count = 0;
	for (size_t i = 0; i < rawDepths.size(); ++i) {
		if (i >= report.inlier_mask.size() || !report.inlier_mask[i]) {
			continue;
		}
		const double predicted = PredictDepth(rawDepths[i], report.model);
		if (!std::isfinite(predicted) || predicted <= kMinDepth) {
			continue;
		}
		const double residual = predicted - metricDepths[i];
		const double relativeResidual = residual / metricDepths[i];
		residualSum += residual;
		squaredSum += residual * residual;
		relativeSum += relativeResidual;
		relativeSquaredSum += relativeResidual * relativeResidual;
		++count;
	}

	if (count < static_cast<size_t>(std::max(DepthAlignmentEstimator::kMinNumSamples, minPoints / 2))) {
		return false;
	}

	*bestModel = report.model;
	*bestRmse = std::sqrt(squaredSum / static_cast<double>(count));
	*bestMeanResidual = residualSum / static_cast<double>(count);
	const double relativeMean = relativeSum / static_cast<double>(count);
	*bestRelativeVariance = std::max(
		0.0,
		relativeSquaredSum / static_cast<double>(count) -
			relativeMean * relativeMean);
	*bestInliers = count;
	return true;
}

DepthModel FitAlignmentModel(
	const std::vector<double>& rawDepths,
	const std::vector<double>& metricDepths,
	const std::string& requestedMode,
	const int minPoints,
	std::string* chosenMode,
	double* chosenRmse,
	double* chosenMeanResidual,
	double* chosenRelativeVariance,
	size_t* chosenInliers)
{
	std::vector<AlignmentMode> modes;
	const std::string modeLower = ToLower(requestedMode);
	if (modeLower == "auto" || modeLower.empty()) {
		modes = {
			AlignmentMode::Linear,
			AlignmentMode::Inverse,
			AlignmentMode::Log,
			AlignmentMode::Quadratic,
			AlignmentMode::InvQuad
		};
	}
	else {
		AlignmentMode mode;
		if (!ParseMode(modeLower, &mode)) {
			throw std::runtime_error("Unsupported depth alignment mode: " + requestedMode);
		}
		modes = { mode };
	}

	DepthModel bestModel;
	double bestRmse = std::numeric_limits<double>::max();
	double bestMeanResidual = 0.0;
	double bestRelativeVariance = 0.0;
	size_t bestInliers = 0;
	std::string bestMode;

	for (const AlignmentMode mode : modes) {
		DepthModel model;
		double rmse = 0.0;
		double meanResidual = 0.0;
		double relativeVariance = 0.0;
		size_t inliers = 0;
		if (!FitMode(
			mode,
			rawDepths,
			metricDepths,
			minPoints,
			&model,
			&rmse,
			&meanResidual,
			&relativeVariance,
			&inliers)) {
			continue;
		}
		if (rmse < bestRmse) {
			bestModel = model;
			bestRmse = rmse;
			bestMeanResidual = meanResidual;
			bestRelativeVariance = relativeVariance;
			bestInliers = inliers;
			bestMode = ModeName(mode);
		}
	}

	if (!bestModel.valid) {
		throw std::runtime_error("RANSAC found no valid depth alignment model.");
	}

	*chosenMode = bestMode;
	*chosenRmse = bestRmse;
	*chosenMeanResidual = bestMeanResidual;
	*chosenRelativeVariance = bestRelativeVariance;
	*chosenInliers = bestInliers;
	return bestModel;
}

float ClipPredictedDepth(const double predicted, const double maxDepth)
{
	if (!std::isfinite(predicted) || predicted <= kMinDepth) {
		return 0.0f;
	}
	return static_cast<float>(
		std::min(std::max(predicted, static_cast<double>(kMinDepth)), maxDepth));
}

NpyFloatImage ApplyModelToImage(
	const NpyFloatImage& rawDepth,
	const DepthModel& model,
	const double maxDepth)
{
	NpyFloatImage aligned = rawDepth;
	for (size_t i = 0; i < aligned.data.size(); ++i) {
		aligned.data[i] = ClipPredictedDepth(
			PredictDepth(static_cast<double>(rawDepth.data[i]), model),
			maxDepth);
	}
	return aligned;
}

std::vector<int> MakeWindowStarts(
	const int length,
	const int grid,
	const double overlap,
	int* windowLength)
{
	if (grid <= 1 || length <= 1) {
		*windowLength = length;
		return { 0 };
	}
	const double denominator =
		1.0 + static_cast<double>(grid - 1) * (1.0 - overlap);
	*windowLength = std::max(
		1,
		std::min(length, static_cast<int>(std::ceil(length / denominator))));
	const int lastStart = std::max(0, length - *windowLength);
	std::vector<int> starts(grid, 0);
	for (int i = 0; i < grid; ++i) {
		starts[i] = static_cast<int>(std::lround(
			static_cast<double>(i) * lastStart / (grid - 1)));
	}
	starts.front() = 0;
	starts.back() = lastStart;
	return starts;
}

double RaisedCosineRamp(const int offset, const int overlapLength, const bool rising)
{
	if (overlapLength <= 0) {
		return 1.0;
	}
	const double t = std::clamp(
		(static_cast<double>(offset) + 0.5) / overlapLength,
		0.0,
		1.0);
	const double risingWeight = 0.5 - 0.5 * std::cos(std::acos(-1.0) * t);
	return rising ? risingWeight : 1.0 - risingWeight;
}

std::vector<double> MakeBlendWindow(
	const int xIndex,
	const int yIndex,
	const std::vector<int>& startsX,
	const std::vector<int>& startsY,
	const int width,
	const int height)
{
	const int leftOverlap = xIndex > 0
		? std::max(0, startsX[xIndex - 1] + width - startsX[xIndex])
		: 0;
	const int rightOverlap = xIndex + 1 < static_cast<int>(startsX.size())
		? std::max(0, startsX[xIndex] + width - startsX[xIndex + 1])
		: 0;
	const int topOverlap = yIndex > 0
		? std::max(0, startsY[yIndex - 1] + height - startsY[yIndex])
		: 0;
	const int bottomOverlap = yIndex + 1 < static_cast<int>(startsY.size())
		? std::max(0, startsY[yIndex] + height - startsY[yIndex + 1])
		: 0;

	std::vector<double> window(static_cast<size_t>(height) * width, 1.0);
	for (int y = 0; y < height; ++y) {
		double wy = 1.0;
		if (topOverlap > 0 && y < topOverlap) {
			wy *= RaisedCosineRamp(y, topOverlap, true);
		}
		if (bottomOverlap > 0 && y >= height - bottomOverlap) {
			wy *= RaisedCosineRamp(
				y - (height - bottomOverlap), bottomOverlap, false);
		}
		for (int x = 0; x < width; ++x) {
			double wx = 1.0;
			if (leftOverlap > 0 && x < leftOverlap) {
				wx *= RaisedCosineRamp(x, leftOverlap, true);
			}
			if (rightOverlap > 0 && x >= width - rightOverlap) {
				wx *= RaisedCosineRamp(
					x - (width - rightOverlap), rightOverlap, false);
			}
			window[static_cast<size_t>(y) * width + x] = wx * wy;
		}
	}
	return window;
}

struct AlignmentWindowWork {
	std::vector<double> blendWindow;
	DepthModel model;
	AlignmentWindowStats stats;
};

bool ComputeWindowResidualStats(
	const std::vector<double>& rawSamples,
	const std::vector<double>& metricSamples,
	const DepthModel& model,
	const std::string& metricName,
	AlignmentWindowStats* stats)
{
	if (!model.valid || rawSamples.size() != metricSamples.size() ||
		rawSamples.empty()) {
		return false;
	}
	double sum = 0.0;
	double squared = 0.0;
	size_t count = 0;
	for (size_t i = 0; i < rawSamples.size(); ++i) {
		const double predicted = PredictDepth(rawSamples[i], model);
		if (!std::isfinite(predicted) || predicted <= kMinDepth ||
			!std::isfinite(metricSamples[i]) || metricSamples[i] <= kMinDepth) {
			continue;
		}
		const double residual =
			(predicted - metricSamples[i]) / metricSamples[i];
		if (!std::isfinite(residual)) {
			continue;
		}
		sum += residual;
		squared += residual * residual;
		++count;
	}
	if (count == 0) {
		return false;
	}
	const double mean = sum / static_cast<double>(count);
	const double rmse = std::sqrt(squared / static_cast<double>(count));
	double variance = 0.0;
	for (size_t i = 0; i < rawSamples.size(); ++i) {
		const double predicted = PredictDepth(rawSamples[i], model);
		if (!std::isfinite(predicted) || predicted <= kMinDepth ||
			!std::isfinite(metricSamples[i]) || metricSamples[i] <= kMinDepth) {
			continue;
		}
		const double residual =
			(predicted - metricSamples[i]) / metricSamples[i];
		if (std::isfinite(residual)) {
			const double centered = residual - mean;
			variance += centered * centered;
		}
	}
	variance /= static_cast<double>(count);
	const double residualMetric = rmse * rmse + variance;
	if (!std::isfinite(rmse) || !std::isfinite(variance) ||
		!std::isfinite(residualMetric)) {
		return false;
	}
	stats->samples = count;
	stats->relativeRmse = rmse;
	stats->relativeVariance = variance;
	stats->fusionMetric = metricName == "RMSE"
		? rmse
		: (metricName == "MSE" ? rmse * rmse : residualMetric);
	return std::isfinite(stats->fusionMetric) && stats->fusionMetric >= 0.0;
}

void CollectBlockSamples(
	const NpyFloatImage& rawDepth,
	const std::vector<float>& sparseMetric,
	const int x0,
	const int y0,
	const int width,
	const int height,
	const int maxSamples,
	std::vector<double>* rawSamples,
	std::vector<double>* metricSamples)
{
	std::vector<double> rawAll;
	std::vector<double> metricAll;
	rawAll.reserve(static_cast<size_t>(width) * height);
	metricAll.reserve(static_cast<size_t>(width) * height);

	for (int y = y0; y < y0 + height; ++y) {
		for (int x = x0; x < x0 + width; ++x) {
			const size_t idx = static_cast<size_t>(y) * rawDepth.width + x;
			const float metric = sparseMetric[idx];
			const float raw = rawDepth.data[idx];
			if (metric > kMinDepth && std::isfinite(raw) && raw > kMinDepth) {
				rawAll.push_back(static_cast<double>(raw));
				metricAll.push_back(static_cast<double>(metric));
			}
		}
	}

	rawSamples->clear();
	metricSamples->clear();
	if (maxSamples > 0 && rawAll.size() > static_cast<size_t>(maxSamples)) {
		rawSamples->reserve(maxSamples);
		metricSamples->reserve(maxSamples);
		for (int i = 0; i < maxSamples; ++i) {
			const size_t idx =
				static_cast<size_t>(i) * rawAll.size() / static_cast<size_t>(maxSamples);
			rawSamples->push_back(rawAll[idx]);
			metricSamples->push_back(metricAll[idx]);
		}
	}
	else {
		*rawSamples = std::move(rawAll);
		*metricSamples = std::move(metricAll);
	}
}

bool TryFitLocalModel(
	const std::vector<double>& rawSamples,
	const std::vector<double>& metricSamples,
	const std::string& globalModeName,
	const int minBlockPoints,
	const std::string& localModelMode,
	DepthModel* localModel,
	std::string* localModeName,
	double* localInlierRatio,
	double* localRelativeVariance)
{
	if (rawSamples.size() < static_cast<size_t>(minBlockPoints)) {
		return false;
	}

	const std::string mode = ToLower(localModelMode);
	try {
		if (mode == "per_block" || mode == "perblock") {
			double rmse = 0.0;
			double meanResidual = 0.0;
			double relativeVariance = 0.0;
			size_t inliers = 0;
			*localModel = FitAlignmentModel(
				rawSamples,
				metricSamples,
				"auto",
				minBlockPoints,
				localModeName,
				&rmse,
				&meanResidual,
				&relativeVariance,
				&inliers);
			*localInlierRatio = rawSamples.empty()
				? 0.0
				: static_cast<double>(inliers) / rawSamples.size();
			*localRelativeVariance = relativeVariance;
			return localModel->valid;
		}

		AlignmentMode globalMode;
		if (!ParseMode(globalModeName, &globalMode)) {
			return false;
		}
		double rmse = 0.0;
		double meanResidual = 0.0;
		double relativeVariance = 0.0;
		size_t inliers = 0;
		if (!FitMode(
			globalMode,
			rawSamples,
			metricSamples,
			minBlockPoints,
			localModel,
			&rmse,
			&meanResidual,
			&relativeVariance,
			&inliers)) {
			return false;
		}
		*localModeName = globalModeName;
		*localInlierRatio = rawSamples.empty()
			? 0.0
			: static_cast<double>(inliers) / rawSamples.size();
		*localRelativeVariance = relativeVariance;
		return localModel->valid;
	}
	catch (...) {
		return false;
	}
}

NpyFloatImage ApplyBlocks(
	const NpyFloatImage& rawDepth,
	const std::vector<float>& sparseMetric,
	const DepthModel& globalModel,
	const std::string& globalModeName,
	const NpyFloatImage& globalAligned,
	const double maxDepth,
	const DepthAlignmentConfig& cfg,
	size_t* localBlocks,
	size_t* fallbackBlocks,
	std::map<std::string, size_t>* localModelCounts,
	const bool collectWindowQuality,
	AlignmentQualityData* windowQuality)
{
	const int grid = std::max(1, cfg.Grid);
	if (grid <= 1) {
		return globalAligned;
	}

	const double overlap =
		std::min(std::max(cfg.Overlap, 0.0), 0.95);
	int blockHeight = 0;
	int blockWidth = 0;
	const std::vector<int> startsY =
		MakeWindowStarts(rawDepth.height, grid, overlap, &blockHeight);
	const std::vector<int> startsX =
		MakeWindowStarts(rawDepth.width, grid, overlap, &blockWidth);
	const int minBlockPoints =
		std::max(cfg.MinBlockPoints, DepthAlignmentEstimator::kMinNumSamples);
	const int maxBlockSamples = std::max(0, cfg.MaxBlockSamples);

	std::vector<double> out(rawDepth.data.size(), 0.0);
	std::vector<double> weight(rawDepth.data.size(), 0.0);
	std::vector<double> blockRawSamples;
	std::vector<double> blockMetricSamples;
	std::vector<AlignmentWindowWork> windows;
	windows.reserve(startsY.size() * startsX.size());

	*localBlocks = 0;
	*fallbackBlocks = 0;
	localModelCounts->clear();

	for (int yi = 0; yi < static_cast<int>(startsY.size()); ++yi) {
		const int y0 = startsY[yi];
		for (int xi = 0; xi < static_cast<int>(startsX.size()); ++xi) {
			const int x0 = startsX[xi];
			AlignmentWindowWork work;
			work.blendWindow = MakeBlendWindow(
				xi, yi, startsX, startsY, blockWidth, blockHeight);
			work.model = globalModel;
			work.stats.row = yi;
			work.stats.col = xi;
			work.stats.x0 = x0;
			work.stats.y0 = y0;
			work.stats.width = blockWidth;
			work.stats.height = blockHeight;
			work.stats.upgradeEligible = false;
			DepthModel currentModel = globalModel;
			std::string currentMode = globalModeName;
			bool usesLocalModel = false;
			double blockInlierRatio = 0.0;
			double blockRelativeVariance = 0.0;

			CollectBlockSamples(
				rawDepth,
				sparseMetric,
				x0,
				y0,
				blockWidth,
				blockHeight,
				maxBlockSamples,
				&blockRawSamples,
				&blockMetricSamples);
			// Keep the observed SfM sample count for every window, including
			// fallback windows that never enter local residual evaluation.
			work.stats.samples = blockRawSamples.size();

			if (cfg.LocalFit &&
				blockRawSamples.size() > static_cast<size_t>(minBlockPoints * 2)) {
				DepthModel localModel;
				std::string localModeName;
				if (TryFitLocalModel(
					blockRawSamples,
					blockMetricSamples,
					globalModeName,
					minBlockPoints,
					cfg.LocalModelMode,
					&localModel,
					&localModeName,
					&blockInlierRatio,
					&blockRelativeVariance)) {
					currentModel = localModel;
					work.model = localModel;
					currentMode = localModeName;
					usesLocalModel = true;
				}
			}

			if (usesLocalModel) {
				++(*localBlocks);
				++(*localModelCounts)[currentMode];
			}
			else {
				++(*fallbackBlocks);
			}
			work.stats.localFitSuccess = usesLocalModel;
			work.stats.historyEligible = usesLocalModel;
			work.stats.fallbackUsed = !usesLocalModel;
			work.stats.localModelMode = currentMode;
			work.stats.inlierRatio = blockInlierRatio;
			if (usesLocalModel && collectWindowQuality) {
				if (!ComputeWindowResidualStats(
					blockRawSamples,
					blockMetricSamples,
					currentModel,
					Config::getInstance().weightedTSDF.GlobalP75Metric,
						&work.stats)) {
					work.stats.localFitSuccess = false;
					work.stats.historyEligible = false;
					work.stats.fallbackUsed = true;
					work.model = globalModel;
					work.stats.localModelMode = globalModeName;
				}
				else {
					// The eligibility gate is based on the collected SfM samples,
					// not on the subset that survived residual validation.
					work.stats.samples = blockRawSamples.size();
					work.stats.upgradeEligible =
						work.stats.samples > static_cast<size_t>(
							Config::getInstance().weightedTSDF.MinUpgradeSamples);
				}
			}
			windows.push_back(std::move(work));
		}
	}

	for (const AlignmentWindowWork& work : windows) {
		const int x0 = work.stats.x0;
		const int y0 = work.stats.y0;
		for (int by = 0; by < blockHeight; ++by) {
			for (int bx = 0; bx < blockWidth; ++bx) {
				const int y = y0 + by;
				const int x = x0 + bx;
				const size_t dstIdx = static_cast<size_t>(y) * rawDepth.width + x;
				const size_t winIdx = static_cast<size_t>(by) * blockWidth + bx;
				float depth = ClipPredictedDepth(
					PredictDepth(static_cast<double>(rawDepth.data[dstIdx]), work.model),
					maxDepth);
				if (depth <= 0.0f) {
					depth = globalAligned.data[dstIdx];
				}
				const double blend = work.blendWindow[winIdx];
				out[dstIdx] += static_cast<double>(depth) * blend;
				weight[dstIdx] += blend;
			}
		}
	}

	if (collectWindowQuality && windowQuality) {
		windowQuality->width = rawDepth.width;
		windowQuality->height = rawDepth.height;
		windowQuality->alignmentMetricNumerator.assign(rawDepth.data.size(), 0.0f);
		windowQuality->alignmentFallbackWeight.assign(rawDepth.data.size(), 0.0f);
		windowQuality->alignmentWeightDenominator.assign(rawDepth.data.size(), 0.0f);
		windowQuality->historyEligibilityValues.assign(rawDepth.data.size(), 0.0f);
		windowQuality->upgradeEligibilityValues.assign(rawDepth.data.size(), 0.0f);
		for (const AlignmentWindowWork& work : windows) {
			const bool metricValid = work.stats.localFitSuccess &&
				std::isfinite(work.stats.fusionMetric);
			const int x0 = work.stats.x0;
			const int y0 = work.stats.y0;
			for (int by = 0; by < blockHeight; ++by) {
				for (int bx = 0; bx < blockWidth; ++bx) {
					const size_t idx = static_cast<size_t>(y0 + by) * rawDepth.width + x0 + bx;
					const float blend = static_cast<float>(
						work.blendWindow[static_cast<size_t>(by) * blockWidth + bx]);
					windowQuality->alignmentWeightDenominator[idx] += blend;
					if (metricValid) {
						windowQuality->alignmentMetricNumerator[idx] +=
							blend * static_cast<float>(work.stats.fusionMetric);
						windowQuality->historyEligibilityValues[idx] = 1.0f;
						if (work.stats.upgradeEligible) {
							windowQuality->upgradeEligibilityValues[idx] = 1.0f;
						}
					}
					else {
						windowQuality->alignmentFallbackWeight[idx] += blend;
					}
				}
			}
		}
		for (const AlignmentWindowWork& work : windows) {
			windowQuality->windows.push_back(work.stats);
			if (work.stats.localFitSuccess &&
				std::isfinite(work.stats.fusionMetric)) {
				windowQuality->windowFusionMetrics.push_back(
					static_cast<float>(work.stats.fusionMetric));
			}
		}
	}

	NpyFloatImage aligned = rawDepth;
	size_t uncoveredPixels = 0;
	for (size_t i = 0; i < aligned.data.size(); ++i) {
		if (weight[i] > 1e-6) {
			aligned.data[i] = static_cast<float>(out[i] / weight[i]);
		}
		else {
			aligned.data[i] = globalAligned.data[i];
			++uncoveredPixels;
		}
	}
	if (uncoveredPixels > 0) {
		LOG(WARNING) << "[DepthAlignment] Local windows left "
			<< uncoveredPixels << " pixels uncovered; global alignment was used.";
	}
	return aligned;
}

} // namespace

void SaveAlignmentWindowMaps(
	const AlignmentQualityData& quality,
	const std::vector<float>& fusionMetric,
	const float globalReference,
	const std::string& alignedDepthPath)
{
	if (quality.width <= 0 || quality.height <= 0) {
		return;
	}
	const size_t expected = static_cast<size_t>(quality.width) * quality.height;
	if (fusionMetric.size() != expected) {
		throw std::runtime_error(
			"Alignment-window fusion metric map size mismatch.");
	}
	const std::string rootDir = ParentPath(ParentPath(alignedDepthPath));
	const char sep = rootDir.find('\\') != std::string::npos ? '\\' : '/';
	const std::string rawDepthPath = rootDir + sep + "np" + sep +
		FileName(alignedDepthPath);
	std::vector<float> weight(expected, 0.0f);
	const float reference = std::max(
		globalReference, std::numeric_limits<float>::epsilon());
	for (size_t i = 0; i < expected; ++i) {
		const float metric = std::max(fusionMetric[i], 0.0f);
		weight[i] = reference / (reference +
			std::max(metric, std::numeric_limits<float>::epsilon()));
	}
	SaveAlignmentFloatMap(
		fusionMetric, quality.width, quality.height,
		rawDepthPath, "_fusion_metric", true);
	SaveAlignmentFloatMap(
		weight, quality.width, quality.height,
		rawDepthPath, "_weight", false);
	if (quality.historyEligibilityValues.size() == expected) {
		SaveAlignmentFloatMap(
			quality.historyEligibilityValues, quality.width, quality.height,
			rawDepthPath, "_history_eligibility", false);
	}
	if (quality.upgradeEligibilityValues.size() == expected) {
		SaveAlignmentFloatMap(
			quality.upgradeEligibilityValues, quality.width, quality.height,
			rawDepthPath, "_upgrade_eligibility", false);
	}
}

void AppendAlignmentWindowDiagnostics(
	const AlignmentQualityData& quality,
	const image_t imageID,
	const std::string& imageName,
	const float globalReference)
{
	const std::string header =
		"timestamp,image_id,image_name,window_row,window_col,x0,y0,width,height,"
		"samples,local_fit_success,fallback_used,local_model_mode,inlier_ratio,"
		"relative_rmse,relative_variance,fusion_metric,history_eligible,"
		"upgrade_eligible,global_p75_metric,global_residual_reference";
	for (const AlignmentWindowStats& window : quality.windows) {
		std::ostringstream row;
		row << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
			<< imageID << ","
			<< GpuDiagnostics::EscapeCsvValue(imageName) << ","
			<< window.row << "," << window.col << ","
			<< window.x0 << "," << window.y0 << ","
			<< window.width << "," << window.height << ","
			<< window.samples << ","
			<< (window.localFitSuccess ? "true" : "false") << ","
			<< (window.fallbackUsed ? "true" : "false") << ","
			<< GpuDiagnostics::EscapeCsvValue(window.localModelMode) << ","
			<< window.inlierRatio << ","
			<< window.relativeRmse << ","
			<< window.relativeVariance << ","
			<< window.fusionMetric << ","
			<< (window.historyEligible ? "true" : "false") << ","
			<< (window.upgradeEligible ? "true" : "false") << ","
			<< quality.globalP75Metric << "," << globalReference;
		GpuDiagnostics::AppendTimingRow(
			"depth_alignment_windows.csv", header, row.str());
	}
}

DepthAlignmentTask::DepthAlignmentTask(
	WorkData* data,
	const image_t imageID,
	const QString& depthPath,
	const QString& colorPath,
	QObject* parent)
	: BaseTask(data, parent)
	, imageID_(imageID)
	, rawDepthPath_(depthPath)
	, colorPath_(colorPath)
{
}

void DepthAlignmentTask::run()
{
	try {
		Timer totalTimer;
		totalTimer.Start();
		const auto& cfg = Config::getInstance().depthAlignment;
		const auto& weightedCfg = Config::getInstance().weightedTSDF;
		const NpyFloatImage rawDepth =
			ReadNpyFloatImage(rawDepthPath_.toStdString());
		if (rawDepth.width <= 0 || rawDepth.height <= 0 || rawDepth.data.empty()) {
			throw std::runtime_error("Raw depth has invalid dimensions.");
		}

		std::vector<float> sparseMetric(
			static_cast<size_t>(rawDepth.width) * rawDepth.height,
			0.0f);
		std::vector<double> sparseTrackLength(sparseMetric.size(), 0.0);
		std::vector<double> sparseReprojectionError(sparseMetric.size(), 2.0);

		{
			QReadLocker locker(&(m_data->m_models_lock[0]));
			if (m_data->getModelManager()->Size() == 0) {
				throw std::runtime_error("No reconstruction is available for depth alignment.");
			}

			const std::shared_ptr<Reconstruction>& model =
				m_data->getModelManager()->Get(0);
			if (!model || !model->ExistsImage(imageID_) ||
				!model->IsImageRegistered(imageID_)) {
				throw std::runtime_error("Image is not registered for depth alignment.");
			}

			const Image& image = model->Image(imageID_);
			if (!model->ExistsCamera(image.CameraId())) {
				throw std::runtime_error("Camera is missing for depth alignment.");
			}
			const Camera& camera = model->Camera(image.CameraId());
			const double scaleX =
				static_cast<double>(rawDepth.width) / std::max<size_t>(camera.Width(), 1u);
			const double scaleY =
				static_cast<double>(rawDepth.height) / std::max<size_t>(camera.Height(), 1u);

			for (const Point2D& point2D : image.Points2D()) {
				if (!point2D.HasPoint3D() ||
					!model->ExistsPoint3D(point2D.point3D_id)) {
					continue;
				}

				const Point3D& point3D = model->Point3D(point2D.point3D_id);
				const Eigen::Vector3d xyzCam =
					image.CamFromWorld() * point3D.XYZ();
				const double z = xyzCam.z();
				if (!std::isfinite(z) || z <= kMinDepth) {
					continue;
				}

				const int x = static_cast<int>(std::lround(point2D.xy.x() * scaleX));
				const int y = static_cast<int>(std::lround(point2D.xy.y() * scaleY));
				if (x < 0 || x >= rawDepth.width || y < 0 || y >= rawDepth.height) {
					continue;
				}

				const size_t sparseIdx =
					static_cast<size_t>(y) * rawDepth.width + x;
				float& sparseValue = sparseMetric[sparseIdx];
				if (sparseValue <= 0.0f || z < sparseValue) {
					sparseValue = static_cast<float>(z);
					sparseTrackLength[sparseIdx] =
						static_cast<double>(point3D.Track().Length());
					sparseReprojectionError[sparseIdx] =
						point3D.HasError() ? point3D.Error() : 2.0;
				}
			}
		}

		SaveAlignmentSparseDepth(
			cfg.SaveAlignDepth, imageID_, rawDepthPath_.toStdString(),
			rawDepth.width, rawDepth.height, sparseMetric);

		std::vector<double> rawSamples;
		std::vector<double> metricSamples;
		std::vector<SparseAlignmentSample> sparseSamples;
		rawSamples.reserve(sparseMetric.size());
		metricSamples.reserve(sparseMetric.size());
		sparseSamples.reserve(sparseMetric.size());
		for (size_t i = 0; i < sparseMetric.size(); ++i) {
			const float metric = sparseMetric[i];
			const float raw = rawDepth.data[i];
			if (metric > kMinDepth &&
				std::isfinite(raw) &&
				raw > kMinDepth) {
			rawSamples.push_back(static_cast<double>(raw));
				metricSamples.push_back(static_cast<double>(metric));
				SparseAlignmentSample sample;
				sample.x = static_cast<int>(i % rawDepth.width);
				sample.y = static_cast<int>(i / rawDepth.width);
				sample.rawDepth = raw;
				sample.metricDepth = metric;
				sample.trackLength = sparseTrackLength[i];
				sample.reprojectionError = sparseReprojectionError[i];
				sparseSamples.push_back(sample);
			}
		}

		const int minPoints = std::max(cfg.MinPoints, DepthAlignmentEstimator::kMinNumSamples);
		if (rawSamples.size() < static_cast<size_t>(minPoints)) {
			throw std::runtime_error(
				"Not enough valid SfM/depth correspondences for depth alignment.");
		}

		std::string chosenMode;
		double rmse = 0.0;
		double meanResidual = 0.0;
		double relativeVariance = 0.0;
		size_t inliers = 0;
		const DepthModel model = FitAlignmentModel(
			rawSamples,
			metricSamples,
			cfg.Mode,
			minPoints,
			&chosenMode,
			&rmse,
			&meanResidual,
			&relativeVariance,
			&inliers);

		const double maxDepth = cfg.MaxDepth > 0.0
			? cfg.MaxDepth
			: Config::getInstance().denseFusion.DepthMax;
		const NpyFloatImage globalAligned =
			ApplyModelToImage(rawDepth, model, maxDepth);
		size_t localBlocks = 0;
		size_t fallbackBlocks = 0;
		std::map<std::string, size_t> localModelCounts;
		const bool collectWindowQuality =
			Config::getInstance().weightedTSDF.Enable &&
			Config::getInstance().denseFusion.Enable;
		AlignmentQualityData windowQuality;
		const NpyFloatImage aligned = ApplyBlocks(
			rawDepth,
			sparseMetric,
			model,
			chosenMode,
			globalAligned,
			maxDepth,
			cfg,
			&localBlocks,
			&fallbackBlocks,
			&localModelCounts,
			collectWindowQuality,
			collectWindowQuality ? &windowQuality : nullptr);

		std::shared_ptr<AlignmentQualityData> qualityData;
		if (collectWindowQuality) {
			qualityData = std::make_shared<AlignmentQualityData>(
				std::move(windowQuality));
			qualityData->globalP75Metric =
				Config::getInstance().weightedTSDF.GlobalP75Metric;
			m_data->StoreAlignmentQuality(imageID_, qualityData);
		}

		const std::string alignedPath =
			MakeAlignedDepthPath(rawDepthPath_.toStdString());
		Timer writeTimer;
		writeTimer.Start();
		WriteNpyFloatImage(alignedPath, aligned);
		const double writeMs =
			static_cast<double>(writeTimer.ElapsedMicroSeconds()) / 1000.0;
		const double totalMs =
			static_cast<double>(totalTimer.ElapsedMicroSeconds()) / 1000.0;

		LOG(INFO) << "[DepthAlignment] Image " << imageID_
			<< " aligned: " << alignedPath
			<< ", mode: " << chosenMode
			<< ", inliers: " << inliers << "/" << rawSamples.size()
			<< ", rmse: " << rmse
			<< ", mean residual: " << meanResidual
			<< ", grid: " << std::max(1, cfg.Grid)
			<< ", local blocks: " << localBlocks
			<< ", fallback blocks: " << fallbackBlocks;
		if (qualityData) {
			LOG(INFO) << "[DepthAlignment] Image " << imageID_
				<< " AlignGrid quality windows: metric="
				<< qualityData->globalP75Metric
				<< ", reliable=" << qualityData->windowFusionMetrics.size()
				<< ", total=" << qualityData->windows.size();
		}
		if (!localModelCounts.empty()) {
			std::ostringstream counts;
			bool first = true;
			for (const auto& item : localModelCounts) {
				if (!first) {
					counts << ", ";
				}
				counts << item.first << "=" << item.second;
				first = false;
			}
			LOG(INFO) << "[DepthAlignment] Image " << imageID_
				<< " local model usage: " << counts.str();
		}
		if (Config::getInstance().pipeline.OutputTimingFiles) {
			std::string imageName = FileName(rawDepthPath_.toStdString());
			const std::string suffix = "_depth.npy";
			if (imageName.size() > suffix.size() &&
				imageName.compare(imageName.size() - suffix.size(), suffix.size(), suffix) == 0) {
				imageName.resize(imageName.size() - suffix.size());
			}
			const std::string header =
				"timestamp,image_id,image_name,mode,chosen_mode,grid,overlap,local_fit,"
				"local_model_mode,num_samples,inliers,inlier_ratio,fit_rmse,"
				"fit_mean_residual,local_blocks,fallback_blocks,alignment_ms,write_ms,"
				"global_p75_metric,reliable_windows,total_windows";
			std::ostringstream row;
			row << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
				<< imageID_ << ","
				<< GpuDiagnostics::EscapeCsvValue(imageName) << ","
				<< GpuDiagnostics::EscapeCsvValue(cfg.Mode) << ","
				<< GpuDiagnostics::EscapeCsvValue(chosenMode) << ","
				<< std::max(1, cfg.Grid) << ","
				<< cfg.Overlap << ","
				<< (cfg.LocalFit ? "true" : "false") << ","
				<< GpuDiagnostics::EscapeCsvValue(cfg.LocalModelMode) << ","
				<< rawSamples.size() << ","
				<< inliers << ","
				<< (rawSamples.empty() ? 0.0 : static_cast<double>(inliers) / rawSamples.size()) << ","
				<< rmse << ","
				<< meanResidual << ","
				<< localBlocks << ","
				<< fallbackBlocks << ","
				<< totalMs << ","
				<< writeMs << ","
				<< GpuDiagnostics::EscapeCsvValue(weightedCfg.GlobalP75Metric) << ","
				<< (qualityData ? qualityData->windowFusionMetrics.size() : 0) << ","
				<< (qualityData ? qualityData->windows.size() : 0);
			GpuDiagnostics::AppendTimingRow(
				"depth_alignment_runtime.csv", header, row.str());

			const std::string qualityHeader =
				"timestamp,method,image_id,image_name,stage,width,height,total_pixels,"
				"valid_ratio,zero_ratio,nan_ratio,inf_ratio,min,max,mean,median,p1,p99";
			auto appendDepthQuality = [&](const char* stage, const NpyFloatImage& depth) {
				size_t finiteCount = 0;
				size_t validCount = 0;
				size_t zeroCount = 0;
				size_t nanCount = 0;
				size_t infCount = 0;
				std::vector<float> values;
				values.reserve(depth.data.size());
				for (const float value : depth.data) {
					if (std::isnan(value)) {
						++nanCount;
						continue;
					}
					if (std::isinf(value)) {
						++infCount;
						continue;
					}
					++finiteCount;
					if (value == 0.0f) {
						++zeroCount;
					}
					if (value > kMinDepth) {
						++validCount;
						values.push_back(value);
					}
				}
				std::sort(values.begin(), values.end());
				const auto percentile = [&](const double p) {
					if (values.empty()) {
						return 0.0;
					}
					const size_t idx = std::min(
						values.size() - 1,
						static_cast<size_t>(std::round(p * (values.size() - 1))));
					return static_cast<double>(values[idx]);
				};
				double mean = 0.0;
				for (const float value : values) {
					mean += value;
				}
				if (!values.empty()) {
					mean /= values.size();
				}
				const double total = static_cast<double>(depth.data.size());
				std::ostringstream qualityRow;
				qualityRow << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
					<< "DepthAlignment,"
					<< imageID_ << ","
					<< GpuDiagnostics::EscapeCsvValue(imageName) << ","
					<< stage << ","
					<< depth.width << ","
					<< depth.height << ","
					<< depth.data.size() << ","
					<< (total > 0.0 ? validCount / total : 0.0) << ","
					<< (total > 0.0 ? zeroCount / total : 0.0) << ","
					<< (total > 0.0 ? nanCount / total : 0.0) << ","
					<< (total > 0.0 ? infCount / total : 0.0) << ","
					<< percentile(0.0) << ","
					<< percentile(1.0) << ","
					<< mean << ","
					<< percentile(0.5) << ","
					<< percentile(0.01) << ","
					<< percentile(0.99);
				GpuDiagnostics::AppendTimingRow(
					"depth_quality_runtime.csv", qualityHeader, qualityRow.str());
			};
			appendDepthQuality("aligned", aligned);
		}
		emit depthAlignmentFinished(
			static_cast<quint32>(imageID_),
			QString::fromStdString(alignedPath),
			colorPath_);
	}
	catch (const std::exception& e) {
		m_data->RemoveAlignmentQuality(imageID_);
		LOG(WARNING) << "[DepthAlignment] Image " << imageID_
			<< " failed: " << e.what();
		emit depthAlignmentFailed(static_cast<quint32>(imageID_), QString::fromStdString(e.what()));
	}
	catch (...) {
		m_data->RemoveAlignmentQuality(imageID_);
		const QString message = "Unknown depth alignment error.";
		LOG(WARNING) << "[DepthAlignment] Image " << imageID_
			<< " failed: " << message.toStdString();
		emit depthAlignmentFailed(static_cast<quint32>(imageID_), message);
	}
}
