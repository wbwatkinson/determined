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


class V1ResourcePoolAwsDetail(object):
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
        'region': 'str',
        'root_volume_size': 'int',
        'image_id': 'str',
        'tag_key': 'str',
        'tag_value': 'str',
        'instance_name': 'str',
        'ssh_key_name': 'str',
        'public_ip': 'bool',
        'subnet_id': 'str',
        'security_group_id': 'str',
        'iam_instance_profile_arn': 'str',
        'instance_type': 'str',
        'log_group': 'str',
        'log_stream': 'str',
        'spot_enabled': 'bool',
        'spot_max_price': 'str',
        'custom_tags': 'list[V1AwsCustomTag]'
    }

    attribute_map = {
        'region': 'region',
        'root_volume_size': 'rootVolumeSize',
        'image_id': 'imageId',
        'tag_key': 'tagKey',
        'tag_value': 'tagValue',
        'instance_name': 'instanceName',
        'ssh_key_name': 'sshKeyName',
        'public_ip': 'publicIp',
        'subnet_id': 'subnetId',
        'security_group_id': 'securityGroupId',
        'iam_instance_profile_arn': 'iamInstanceProfileArn',
        'instance_type': 'instanceType',
        'log_group': 'logGroup',
        'log_stream': 'logStream',
        'spot_enabled': 'spotEnabled',
        'spot_max_price': 'spotMaxPrice',
        'custom_tags': 'customTags'
    }

    def __init__(self, region=None, root_volume_size=None, image_id=None, tag_key=None, tag_value=None, instance_name=None, ssh_key_name=None, public_ip=None, subnet_id=None, security_group_id=None, iam_instance_profile_arn=None, instance_type=None, log_group=None, log_stream=None, spot_enabled=None, spot_max_price=None, custom_tags=None):  # noqa: E501
        """V1ResourcePoolAwsDetail - a model defined in Swagger"""  # noqa: E501

        self._region = None
        self._root_volume_size = None
        self._image_id = None
        self._tag_key = None
        self._tag_value = None
        self._instance_name = None
        self._ssh_key_name = None
        self._public_ip = None
        self._subnet_id = None
        self._security_group_id = None
        self._iam_instance_profile_arn = None
        self._instance_type = None
        self._log_group = None
        self._log_stream = None
        self._spot_enabled = None
        self._spot_max_price = None
        self._custom_tags = None
        self.discriminator = None

        self.region = region
        self.root_volume_size = root_volume_size
        self.image_id = image_id
        self.tag_key = tag_key
        self.tag_value = tag_value
        self.instance_name = instance_name
        self.ssh_key_name = ssh_key_name
        self.public_ip = public_ip
        if subnet_id is not None:
            self.subnet_id = subnet_id
        self.security_group_id = security_group_id
        self.iam_instance_profile_arn = iam_instance_profile_arn
        if instance_type is not None:
            self.instance_type = instance_type
        if log_group is not None:
            self.log_group = log_group
        if log_stream is not None:
            self.log_stream = log_stream
        self.spot_enabled = spot_enabled
        if spot_max_price is not None:
            self.spot_max_price = spot_max_price
        if custom_tags is not None:
            self.custom_tags = custom_tags

    @property
    def region(self):
        """Gets the region of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The region of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._region

    @region.setter
    def region(self, region):
        """Sets the region of this V1ResourcePoolAwsDetail.


        :param region: The region of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if region is None:
            raise ValueError("Invalid value for `region`, must not be `None`")  # noqa: E501

        self._region = region

    @property
    def root_volume_size(self):
        """Gets the root_volume_size of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The root_volume_size of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: int
        """
        return self._root_volume_size

    @root_volume_size.setter
    def root_volume_size(self, root_volume_size):
        """Sets the root_volume_size of this V1ResourcePoolAwsDetail.


        :param root_volume_size: The root_volume_size of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: int
        """
        if root_volume_size is None:
            raise ValueError("Invalid value for `root_volume_size`, must not be `None`")  # noqa: E501

        self._root_volume_size = root_volume_size

    @property
    def image_id(self):
        """Gets the image_id of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The image_id of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._image_id

    @image_id.setter
    def image_id(self, image_id):
        """Sets the image_id of this V1ResourcePoolAwsDetail.


        :param image_id: The image_id of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if image_id is None:
            raise ValueError("Invalid value for `image_id`, must not be `None`")  # noqa: E501

        self._image_id = image_id

    @property
    def tag_key(self):
        """Gets the tag_key of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The tag_key of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._tag_key

    @tag_key.setter
    def tag_key(self, tag_key):
        """Sets the tag_key of this V1ResourcePoolAwsDetail.


        :param tag_key: The tag_key of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if tag_key is None:
            raise ValueError("Invalid value for `tag_key`, must not be `None`")  # noqa: E501

        self._tag_key = tag_key

    @property
    def tag_value(self):
        """Gets the tag_value of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The tag_value of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._tag_value

    @tag_value.setter
    def tag_value(self, tag_value):
        """Sets the tag_value of this V1ResourcePoolAwsDetail.


        :param tag_value: The tag_value of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if tag_value is None:
            raise ValueError("Invalid value for `tag_value`, must not be `None`")  # noqa: E501

        self._tag_value = tag_value

    @property
    def instance_name(self):
        """Gets the instance_name of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The instance_name of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._instance_name

    @instance_name.setter
    def instance_name(self, instance_name):
        """Sets the instance_name of this V1ResourcePoolAwsDetail.


        :param instance_name: The instance_name of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if instance_name is None:
            raise ValueError("Invalid value for `instance_name`, must not be `None`")  # noqa: E501

        self._instance_name = instance_name

    @property
    def ssh_key_name(self):
        """Gets the ssh_key_name of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The ssh_key_name of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._ssh_key_name

    @ssh_key_name.setter
    def ssh_key_name(self, ssh_key_name):
        """Sets the ssh_key_name of this V1ResourcePoolAwsDetail.


        :param ssh_key_name: The ssh_key_name of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if ssh_key_name is None:
            raise ValueError("Invalid value for `ssh_key_name`, must not be `None`")  # noqa: E501

        self._ssh_key_name = ssh_key_name

    @property
    def public_ip(self):
        """Gets the public_ip of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The public_ip of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: bool
        """
        return self._public_ip

    @public_ip.setter
    def public_ip(self, public_ip):
        """Sets the public_ip of this V1ResourcePoolAwsDetail.


        :param public_ip: The public_ip of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: bool
        """
        if public_ip is None:
            raise ValueError("Invalid value for `public_ip`, must not be `None`")  # noqa: E501

        self._public_ip = public_ip

    @property
    def subnet_id(self):
        """Gets the subnet_id of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The subnet_id of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._subnet_id

    @subnet_id.setter
    def subnet_id(self, subnet_id):
        """Sets the subnet_id of this V1ResourcePoolAwsDetail.


        :param subnet_id: The subnet_id of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """

        self._subnet_id = subnet_id

    @property
    def security_group_id(self):
        """Gets the security_group_id of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The security_group_id of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._security_group_id

    @security_group_id.setter
    def security_group_id(self, security_group_id):
        """Sets the security_group_id of this V1ResourcePoolAwsDetail.


        :param security_group_id: The security_group_id of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if security_group_id is None:
            raise ValueError("Invalid value for `security_group_id`, must not be `None`")  # noqa: E501

        self._security_group_id = security_group_id

    @property
    def iam_instance_profile_arn(self):
        """Gets the iam_instance_profile_arn of this V1ResourcePoolAwsDetail.  # noqa: E501

        The Amazon Resource Name (ARN) of the IAM instance profile to attach to the agent instances.  # noqa: E501

        :return: The iam_instance_profile_arn of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._iam_instance_profile_arn

    @iam_instance_profile_arn.setter
    def iam_instance_profile_arn(self, iam_instance_profile_arn):
        """Sets the iam_instance_profile_arn of this V1ResourcePoolAwsDetail.

        The Amazon Resource Name (ARN) of the IAM instance profile to attach to the agent instances.  # noqa: E501

        :param iam_instance_profile_arn: The iam_instance_profile_arn of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """
        if iam_instance_profile_arn is None:
            raise ValueError("Invalid value for `iam_instance_profile_arn`, must not be `None`")  # noqa: E501

        self._iam_instance_profile_arn = iam_instance_profile_arn

    @property
    def instance_type(self):
        """Gets the instance_type of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The instance_type of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._instance_type

    @instance_type.setter
    def instance_type(self, instance_type):
        """Sets the instance_type of this V1ResourcePoolAwsDetail.


        :param instance_type: The instance_type of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """

        self._instance_type = instance_type

    @property
    def log_group(self):
        """Gets the log_group of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The log_group of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._log_group

    @log_group.setter
    def log_group(self, log_group):
        """Sets the log_group of this V1ResourcePoolAwsDetail.


        :param log_group: The log_group of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """

        self._log_group = log_group

    @property
    def log_stream(self):
        """Gets the log_stream of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The log_stream of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._log_stream

    @log_stream.setter
    def log_stream(self, log_stream):
        """Sets the log_stream of this V1ResourcePoolAwsDetail.


        :param log_stream: The log_stream of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """

        self._log_stream = log_stream

    @property
    def spot_enabled(self):
        """Gets the spot_enabled of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The spot_enabled of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: bool
        """
        return self._spot_enabled

    @spot_enabled.setter
    def spot_enabled(self, spot_enabled):
        """Sets the spot_enabled of this V1ResourcePoolAwsDetail.


        :param spot_enabled: The spot_enabled of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: bool
        """
        if spot_enabled is None:
            raise ValueError("Invalid value for `spot_enabled`, must not be `None`")  # noqa: E501

        self._spot_enabled = spot_enabled

    @property
    def spot_max_price(self):
        """Gets the spot_max_price of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The spot_max_price of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: str
        """
        return self._spot_max_price

    @spot_max_price.setter
    def spot_max_price(self, spot_max_price):
        """Sets the spot_max_price of this V1ResourcePoolAwsDetail.


        :param spot_max_price: The spot_max_price of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: str
        """

        self._spot_max_price = spot_max_price

    @property
    def custom_tags(self):
        """Gets the custom_tags of this V1ResourcePoolAwsDetail.  # noqa: E501


        :return: The custom_tags of this V1ResourcePoolAwsDetail.  # noqa: E501
        :rtype: list[V1AwsCustomTag]
        """
        return self._custom_tags

    @custom_tags.setter
    def custom_tags(self, custom_tags):
        """Sets the custom_tags of this V1ResourcePoolAwsDetail.


        :param custom_tags: The custom_tags of this V1ResourcePoolAwsDetail.  # noqa: E501
        :type: list[V1AwsCustomTag]
        """

        self._custom_tags = custom_tags

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
        if issubclass(V1ResourcePoolAwsDetail, dict):
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
        if not isinstance(other, V1ResourcePoolAwsDetail):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
