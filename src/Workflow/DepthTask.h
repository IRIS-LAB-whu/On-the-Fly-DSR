#pragma once

#include "BaseTask.h"
#include "../Base/types.h"
#include <QString>

class DepthTask : public BaseTask
{
	Q_OBJECT

public:
	enum class Mode {
		Warmup,
		Predict
	};

	explicit DepthTask(WorkData* data,
		QObject* parent = nullptr)
		: BaseTask(data, parent)
		, mode(Mode::Warmup)
		, imageID(kInvalidImageId) {}

	explicit DepthTask(WorkData* data,
		image_t imageID,
		QObject* parent = nullptr)
		: BaseTask(data, parent)
		, mode(Mode::Predict)
		, imageID(imageID) {}

	void run() override;

signals:
	void depthWorkerInitialized(QString message);
	void depthWorkerInitializeFailed(QString errorMessage);
	void depthPredictionFinished(quint32 imageID, QString depthPath, QString colorPath);
	void depthPredictionFailed(quint32 imageID, QString errorMessage);

private:
	Mode mode;
	image_t imageID;
};
