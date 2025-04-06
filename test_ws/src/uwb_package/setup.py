from setuptools import find_packages, setup

package_name = 'uwb_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Roman Eidelman, Artem Kondaratev',
    maintainer_email='reiv.dev@gmail.com, artemkondratev5@gmail.com',
    description='Package for streaming UWB coordinates',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'uwb_streaming = uwb_package.uwb_streaming:main'
        ],
    },
)
