#pragma once

#include "BaseTask.h"
#include <QString>

class DenseScaleInitTask : public BaseTask
{
	Q_OBJECT

public:
	explicit DenseScaleInitTask(WorkData* data, QObject* parent = nullptr)
		: BaseTask(data, parent) {}

	void run() override;

signals:
	void denseScaleInitialized(bool success, QString message);
};
