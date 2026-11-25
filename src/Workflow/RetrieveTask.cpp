#include "../Base/timer.h"
#include "RetrieveTask.h"
#include "WorkData.h"

#include "glog/logging.h"

void RetrieveTask::run()
{
	image_t imageID = kInvalidImageId;
	bool success = false;
	bool hasPairs = false;
	try {
		std::string imagePathstr = imagePath.toStdString();
		Timer retrievalTimer;
		retrievalTimer.Start();
		imageID = m_data->getImageID(imagePathstr);
		CGlobalFeatureRetriever* globalFeatureRetriever = m_data->getGlobalFeatureRetriever();
		if (globalFeatureRetriever == nullptr) {
			throw std::runtime_error("Retriever is not initialized");
		}

		const std::vector<image_t> retrievalResult = globalFeatureRetriever->Retrieve(imageID);

		const size_t retrievalTime = retrievalTimer.ElapsedMicroSeconds() / 1000;
		m_data->globalFeatureRetrievalTimes[imageID] = retrievalTime;

		std::string output = "[Retrieval] Result: " + m_data->DatabaseReadImage(imageID).Name() + " -> ";
		for (image_t i = 0; i < retrievalResult.size(); i++)
		{
			output += m_data->DatabaseReadImage(retrievalResult[i]).Name();
			if (static_cast<size_t>(i) + 2 <= retrievalResult.size())
			{
				output += ", ";
			}
		}
		LOG(INFO) << output;

		m_data->numMatchingPairs[imageID] = retrievalResult.size();
		hasPairs = !retrievalResult.empty();
		if (hasPairs)
		{
			m_data->setImageStatus(imagePathstr, CImageStatusFlag::CMatching);
		}
		else
		{
			m_data->setImageStatus(imagePathstr, CImageStatusFlag::CMatched);
		}
		for (const auto target : retrievalResult) emit retrieveTaskFinished(target, imageID);
		success = true;
	}
	catch (const std::exception& e) {
		LOG(ERROR) << "[Retrieval] Image " << imageID << " failed: " << e.what();
	}
	catch (...) {
		LOG(ERROR) << "[Retrieval] Image " << imageID << " failed with unknown exception.";
	}
	emit retrievalCompleted(imageID, success, hasPairs);
	if (!success) emit frontEndFailed(imagePath, "retrieval_failed");
}
