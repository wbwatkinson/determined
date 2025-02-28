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


class V1GetTrialResponse(object):
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
        'trial': 'Trialv1Trial',
        'workloads': 'list[GetTrialResponseWorkloadContainer]'
    }

    attribute_map = {
        'trial': 'trial',
        'workloads': 'workloads'
    }

    def __init__(self, trial=None, workloads=None):  # noqa: E501
        """V1GetTrialResponse - a model defined in Swagger"""  # noqa: E501

        self._trial = None
        self._workloads = None
        self.discriminator = None

        self.trial = trial
        if workloads is not None:
            self.workloads = workloads

    @property
    def trial(self):
        """Gets the trial of this V1GetTrialResponse.  # noqa: E501

        The requested trial.  # noqa: E501

        :return: The trial of this V1GetTrialResponse.  # noqa: E501
        :rtype: Trialv1Trial
        """
        return self._trial

    @trial.setter
    def trial(self, trial):
        """Sets the trial of this V1GetTrialResponse.

        The requested trial.  # noqa: E501

        :param trial: The trial of this V1GetTrialResponse.  # noqa: E501
        :type: Trialv1Trial
        """
        if trial is None:
            raise ValueError("Invalid value for `trial`, must not be `None`")  # noqa: E501

        self._trial = trial

    @property
    def workloads(self):
        """Gets the workloads of this V1GetTrialResponse.  # noqa: E501

        Trial workloads.  # noqa: E501

        :return: The workloads of this V1GetTrialResponse.  # noqa: E501
        :rtype: list[GetTrialResponseWorkloadContainer]
        """
        return self._workloads

    @workloads.setter
    def workloads(self, workloads):
        """Sets the workloads of this V1GetTrialResponse.

        Trial workloads.  # noqa: E501

        :param workloads: The workloads of this V1GetTrialResponse.  # noqa: E501
        :type: list[GetTrialResponseWorkloadContainer]
        """

        self._workloads = workloads

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
        if issubclass(V1GetTrialResponse, dict):
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
        if not isinstance(other, V1GetTrialResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
