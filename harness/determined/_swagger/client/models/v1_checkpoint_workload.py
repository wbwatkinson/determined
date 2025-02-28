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


class V1CheckpointWorkload(object):
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
        'uuid': 'str',
        'start_time': 'datetime',
        'end_time': 'datetime',
        'state': 'Determinedcheckpointv1State',
        'resources': 'dict(str, str)',
        'total_batches': 'int'
    }

    attribute_map = {
        'uuid': 'uuid',
        'start_time': 'startTime',
        'end_time': 'endTime',
        'state': 'state',
        'resources': 'resources',
        'total_batches': 'totalBatches'
    }

    def __init__(self, uuid=None, start_time=None, end_time=None, state=None, resources=None, total_batches=None):  # noqa: E501
        """V1CheckpointWorkload - a model defined in Swagger"""  # noqa: E501

        self._uuid = None
        self._start_time = None
        self._end_time = None
        self._state = None
        self._resources = None
        self._total_batches = None
        self.discriminator = None

        if uuid is not None:
            self.uuid = uuid
        self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        self.state = state
        if resources is not None:
            self.resources = resources
        self.total_batches = total_batches

    @property
    def uuid(self):
        """Gets the uuid of this V1CheckpointWorkload.  # noqa: E501

        UUID of the checkpoint.  # noqa: E501

        :return: The uuid of this V1CheckpointWorkload.  # noqa: E501
        :rtype: str
        """
        return self._uuid

    @uuid.setter
    def uuid(self, uuid):
        """Sets the uuid of this V1CheckpointWorkload.

        UUID of the checkpoint.  # noqa: E501

        :param uuid: The uuid of this V1CheckpointWorkload.  # noqa: E501
        :type: str
        """

        self._uuid = uuid

    @property
    def start_time(self):
        """Gets the start_time of this V1CheckpointWorkload.  # noqa: E501

        The time the workload was started.  # noqa: E501

        :return: The start_time of this V1CheckpointWorkload.  # noqa: E501
        :rtype: datetime
        """
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        """Sets the start_time of this V1CheckpointWorkload.

        The time the workload was started.  # noqa: E501

        :param start_time: The start_time of this V1CheckpointWorkload.  # noqa: E501
        :type: datetime
        """
        if start_time is None:
            raise ValueError("Invalid value for `start_time`, must not be `None`")  # noqa: E501

        self._start_time = start_time

    @property
    def end_time(self):
        """Gets the end_time of this V1CheckpointWorkload.  # noqa: E501

        The time the workload finished or was stopped.  # noqa: E501

        :return: The end_time of this V1CheckpointWorkload.  # noqa: E501
        :rtype: datetime
        """
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        """Sets the end_time of this V1CheckpointWorkload.

        The time the workload finished or was stopped.  # noqa: E501

        :param end_time: The end_time of this V1CheckpointWorkload.  # noqa: E501
        :type: datetime
        """

        self._end_time = end_time

    @property
    def state(self):
        """Gets the state of this V1CheckpointWorkload.  # noqa: E501

        The state of the checkpoint.  # noqa: E501

        :return: The state of this V1CheckpointWorkload.  # noqa: E501
        :rtype: Determinedcheckpointv1State
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this V1CheckpointWorkload.

        The state of the checkpoint.  # noqa: E501

        :param state: The state of this V1CheckpointWorkload.  # noqa: E501
        :type: Determinedcheckpointv1State
        """
        if state is None:
            raise ValueError("Invalid value for `state`, must not be `None`")  # noqa: E501

        self._state = state

    @property
    def resources(self):
        """Gets the resources of this V1CheckpointWorkload.  # noqa: E501

        Dictionary of file paths to file sizes in bytes of all files in the checkpoint.  # noqa: E501

        :return: The resources of this V1CheckpointWorkload.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._resources

    @resources.setter
    def resources(self, resources):
        """Sets the resources of this V1CheckpointWorkload.

        Dictionary of file paths to file sizes in bytes of all files in the checkpoint.  # noqa: E501

        :param resources: The resources of this V1CheckpointWorkload.  # noqa: E501
        :type: dict(str, str)
        """

        self._resources = resources

    @property
    def total_batches(self):
        """Gets the total_batches of this V1CheckpointWorkload.  # noqa: E501

        Total number of batches as of this workload's completion.  # noqa: E501

        :return: The total_batches of this V1CheckpointWorkload.  # noqa: E501
        :rtype: int
        """
        return self._total_batches

    @total_batches.setter
    def total_batches(self, total_batches):
        """Sets the total_batches of this V1CheckpointWorkload.

        Total number of batches as of this workload's completion.  # noqa: E501

        :param total_batches: The total_batches of this V1CheckpointWorkload.  # noqa: E501
        :type: int
        """
        if total_batches is None:
            raise ValueError("Invalid value for `total_batches`, must not be `None`")  # noqa: E501

        self._total_batches = total_batches

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
        if issubclass(V1CheckpointWorkload, dict):
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
        if not isinstance(other, V1CheckpointWorkload):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
