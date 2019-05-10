from setuptools import setup

setup(
    name='mock_atm',
    version='0.0.0',
    packages=[],
    py_modules=['mock_atm_pub'],
    install_requires=['setuptools'],
    author='test',
    author_email='test@hopetechinik.com',
    maintainer='test',
    maintainer_email='test@hopetechinik.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Package containing examples of how to use the DYNAMIXEL ROS 2 API.',
    license='Apache License, Version 2.0',
    test_suite='test',
    entry_points={
        'console_scripts': [
            'mock_atm_pub = mock_atm_pub:main'
        ],
    },
)
