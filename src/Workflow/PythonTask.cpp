#include "../Base/timer.h"
#include "../Base/types.h"
#include "../Feature/PyLoader.h"
#include "glog/logging.h"
#include "PythonTask.h"
#include "WorkData.h"

void PythonTask::run()
{
	try {
		PythonThreadLocker lock;
		Timer globalFeatureExtractionTimer;
		globalFeatureExtractionTimer.Start();
		std::string imagePath = imagePathQstr.toStdString();
		GlobalFeature globalFeature;
		CGlobalFeatureExtractor* globalFeatureExtractor = m_data->getGlobalFeatureExtractor();
		if (globalFeatureExtractor == nullptr) {
			LOG(ERROR) << "[Global feature] Extractor is not initialized.";
			emit frontEndFailed(imagePathQstr, "global_extractor_unavailable");
			return;
		}
		bool extractFlag = globalFeatureExtractor->Extract(imagePath, globalFeature);
		if (extractFlag)
		{
			const size_t imageID = m_data->getImageID(imagePath);
			m_data->DatabaseWriteGlobalFeature(imageID, globalFeature);
			emit globalFeatureExtractFinished(imagePathQstr);

			const size_t globalFeatureExtractionTime = globalFeatureExtractionTimer.ElapsedMicroSeconds() / 1000;
			m_data->globalFeatureExtractionTimes[imageID] = globalFeatureExtractionTime;

			LOG(INFO) << "[Global feature] Image " << imagePath << " extracted successed!";
		}
		else
		{
			LOG(INFO) << "[Global feature] Image " << imagePath << " extracted failed!";
			emit frontEndFailed(imagePathQstr, "global_extraction_failed");
		}
	}
	catch (const std::exception& e) {
		LOG(ERROR) << "[Global feature] Task failed: " << e.what();
		emit frontEndFailed(imagePathQstr, QString::fromUtf8(e.what()));
	}
	catch (...) {
		LOG(ERROR) << "[Global feature] Task failed with unknown exception.";
		emit frontEndFailed(imagePathQstr, "global_extraction_exception");
	}
}
