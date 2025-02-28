# coding: utf-8

"""
    Determined API (Beta)

    Determined helps deep learning teams train models more quickly, easily share GPU resources, and effectively collaborate. Determined allows deep learning engineers to focus on building and training models at scale, without needing to worry about DevOps or writing custom code for common tasks like fault tolerance or experiment tracking.  You can think of Determined as a platform that bridges the gap between tools like TensorFlow and PyTorch --- which work great for a single researcher with a single GPU --- to the challenges that arise when doing deep learning at scale, as teams, clusters, and data sets all increase in size.  # noqa: E501

    OpenAPI spec version: 0.1
    Contact: community@determined.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six


class V1TrainingMetrics(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'trial_id': 'int',
        'trial_run_id': 'int',
        'total_batches': 'int',
        'total_records': 'int',
        'total_epochs': 'float',
        'metrics': 'object',
        'batch_metrics': 'list[object]'
    }

    attribute_map = {
        'trial_id': 'trialId',
        'trial_run_id': 'trialRunId',
        'total_batches': 'totalBatches',
        'total_records': 'totalRecords',
        'total_epochs': 'totalEpochs',
        'metrics': 'metrics',
        'batch_metrics': 'batchMetrics'
    }

    def __init__(self, trial_id=None, trial_run_id=None, total_batches=None, total_records=None, total_epochs=None, metrics=None, batch_metrics=None):  # noqa: E501
        """V1TrainingMetrics - a model defined in Swagger"""  # noqa: E501

        self._trial_id = None
        self._trial_run_id = None
        self._total_batches = None
        self._total_records = None
        self._total_epochs = None
        self._metrics = None
        self._batch_metrics = None
        self.discriminator = None

        self.trial_id = trial_id
        self.trial_run_id = trial_run_id
        self.total_batches = total_batches
        if total_records is not None:
            self.total_records = total_records
        if total_epochs is not None:
            self.total_epochs = total_epochs
        self.metrics = metrics
        if batch_metrics is not None:
            self.batch_metrics = batch_metrics

    @property
    def trial_id(self):
        """Gets the trial_id of this V1TrainingMetrics.  # noqa: E501

        The trial associated with these metrics.  # noqa: E501

        :return: The trial_id of this V1TrainingMetrics.  # noqa: E501
        :rtype: int
        """
        return self._trial_id

    @trial_id.setter
    def trial_id(self, trial_id):
        """Sets the trial_id of this V1TrainingMetrics.

        The trial associated with these metrics.  # noqa: E501

        :param trial_id: The trial_id of this V1TrainingMetrics.  # noqa: E501
        :type: int
        """
        if trial_id is None:
            raise ValueError("Invalid value for `trial_id`, must not be `None`")  # noqa: E501

        self._trial_id = trial_id

    @property
    def trial_run_id(self):
        """Gets the trial_run_id of this V1TrainingMetrics.  # noqa: E501

        The trial run associated with these metrics.  # noqa: E501

        :return: The trial_run_id of this V1TrainingMetrics.  # noqa: E501
        :rtype: int
        """
        return self._trial_run_id

    @trial_run_id.setter
    def trial_run_id(self, trial_run_id):
        """Sets the trial_run_id of this V1TrainingMetrics.

        The trial run associated with these metrics.  # noqa: E501

        :param trial_run_id: The trial_run_id of this V1TrainingMetrics.  # noqa: E501
        :type: int
        """
        if trial_run_id is None:
            raise ValueError("Invalid value for `trial_run_id`, must not be `None`")  # noqa: E501

        self._trial_run_id = trial_run_id

    @property
    def total_batches(self):
        """Gets the total_batches of this V1TrainingMetrics.  # noqa: E501

        The number of batches trained on when these metrics were reported.  # noqa: E501

        :return: The total_batches of this V1TrainingMetrics.  # noqa: E501
        :rtype: int
        """
        return self._total_batches

    @total_batches.setter
    def total_batches(self, total_batches):
        """Sets the total_batches of this V1TrainingMetrics.

        The number of batches trained on when these metrics were reported.  # noqa: E501

        :param total_batches: The total_batches of this V1TrainingMetrics.  # noqa: E501
        :type: int
        """
        if total_batches is None:
            raise ValueError("Invalid value for `total_batches`, must not be `None`")  # noqa: E501

        self._total_batches = total_batches

    @property
    def total_records(self):
        """Gets the total_records of this V1TrainingMetrics.  # noqa: E501

        The number of batches trained on when these metrics were reported.  # noqa: E501

        :return: The total_records of this V1TrainingMetrics.  # noqa: E501
        :rtype: int
        """
        return self._total_records

    @total_records.setter
    def total_records(self, total_records):
        """Sets the total_records of this V1TrainingMetrics.

        The number of batches trained on when these metrics were reported.  # noqa: E501

        :param total_records: The total_records of this V1TrainingMetrics.  # noqa: E501
        :type: int
        """

        self._total_records = total_records

    @property
    def total_epochs(self):
        """Gets the total_epochs of this V1TrainingMetrics.  # noqa: E501

        The number of epochs trained on when these metrics were reported.  # noqa: E501

        :return: The total_epochs of this V1TrainingMetrics.  # noqa: E501
        :rtype: float
        """
        return self._total_epochs

    @total_epochs.setter
    def total_epochs(self, total_epochs):
        """Sets the total_epochs of this V1TrainingMetrics.

        The number of epochs trained on when these metrics were reported.  # noqa: E501

        :param total_epochs: The total_epochs of this V1TrainingMetrics.  # noqa: E501
        :type: float
        """

        self._total_epochs = total_epochs

    @property
    def metrics(self):
        """Gets the metrics of this V1TrainingMetrics.  # noqa: E501

        The metrics for this bit of training (reduced over the reporting period).  # noqa: E501

        :return: The metrics of this V1TrainingMetrics.  # noqa: E501
        :rtype: object
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """Sets the metrics of this V1TrainingMetrics.

        The metrics for this bit of training (reduced over the reporting period).  # noqa: E501

        :param metrics: The metrics of this V1TrainingMetrics.  # noqa: E501
        :type: object
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")  # noqa: E501

        self._metrics = metrics

    @property
    def batch_metrics(self):
        """Gets the batch_metrics of this V1TrainingMetrics.  # noqa: E501

        The batch metrics for this bit of training.  # noqa: E501

        :return: The batch_metrics of this V1TrainingMetrics.  # noqa: E501
        :rtype: list[object]
        """
        return self._batch_metrics

    @batch_metrics.setter
    def batch_metrics(self, batch_metrics):
        """Sets the batch_metrics of this V1TrainingMetrics.

        The batch metrics for this bit of training.  # noqa: E501

        :param batch_metrics: The batch_metrics of this V1TrainingMetrics.  # noqa: E501
        :type: list[object]
        """

        self._batch_metrics = batch_metrics

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(V1TrainingMetrics, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1TrainingMetrics):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
