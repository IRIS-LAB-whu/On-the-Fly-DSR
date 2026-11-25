#include "../Base/timer.h"
#include "../Estimator/IncrementalMapper.h"
#include "DenseScaleInitTask.h"
#include "WorkData.h"

#include <QString>
#include <sstream>

namespace {

std::string FormatDenseScaleMessage(
	const bool baSuccess,
	const size_t numRegImages,
	const size_t numPoints3D,
	const double finalCost,
	const size_t elapsedMs)
{
	std::ostringstream stream;
	stream << "Dense scale initialized with global BA + Normalize(extent=10). "
		<< "BA success=" << (baSuccess ? "true" : "false")
		<< ", registered images=" << numRegImages
		<< ", points=" << numPoints3D
		<< ", final cost=" << finalCost << "px"
		<< ", elapsed=" << elapsedMs << "ms";
	if (!baSuccess) {
		stream << ". Global BA failed; fallback Normalize() was applied.";
	}
	return stream.str();
}

} // namespace

void DenseScaleInitTask::run()
{
	try {
		Timer timer;
		timer.Start();

		IncrementalMapper* mapper = m_data->getIncrementalMapper();
		ReconstructionManager* modelManager = m_data->getModelManager();
		if (mapper == nullptr || modelManager == nullptr || modelManager->Size() == 0) {
			emit denseScaleInitialized(false, "Dense scale initialization failed: reconstruction is not available.");
			return;
		}

		bool baSuccess = false;
		double finalCost = 0.0;
		size_t numRegImages = 0;
		size_t numPoints3D = 0;

		{
			QWriteLocker locker(&(m_data->m_models_lock[0]));
			std::shared_ptr<Reconstruction> model = modelManager->Get(0);
			if (!model) {
				emit denseScaleInitialized(false, "Dense scale initialization failed: model 0 is empty.");
				return;
			}
			if (model->NumRegImages() < 2) {
				emit denseScaleInitialized(false, "Dense scale initialization failed: fewer than 2 registered images.");
				return;
			}

			mapper->BeginReconstruction(model);

			IncrementalMapper::Options mapperOptions;
			BundleAdjustmentOptions baOptions;
			baOptions.ba_type = "DenseScaleInitGlobal BA";
			baOptions.solver_options.function_tolerance = 0.0;
			baOptions.solver_options.gradient_tolerance = 1.0;
			baOptions.solver_options.parameter_tolerance = 0.0;
			baOptions.solver_options.max_num_iterations = 50;
			baOptions.solver_options.max_linear_solver_iterations = 100;
			baOptions.solver_options.minimizer_progress_to_stdout = false;
			baOptions.solver_options.logging_type = ceres::LoggingType::SILENT;

			baSuccess = mapper->AdjustGlobalBundle(mapperOptions, baOptions, finalCost);
			if (!baSuccess) {
				model->Normalize();
			}
			mapper->EndReconstruction(false);

			numRegImages = model->NumRegImages();
			numPoints3D = model->NumPoints3D();
		}

		const size_t elapsedMs = timer.ElapsedMicroSeconds() / 1000;
		emit denseScaleInitialized(
			true,
			QString::fromStdString(FormatDenseScaleMessage(
				baSuccess, numRegImages, numPoints3D, finalCost, elapsedMs)));
	}
	catch (const std::exception& e) {
		emit denseScaleInitialized(false, QString("Dense scale initialization failed: %1").arg(e.what()));
	}
	catch (...) {
		emit denseScaleInitialized(false, "Dense scale initialization failed: unknown exception.");
	}
}
