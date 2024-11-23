from setuptools import find_packages, setup
from glob import glob

package_name = 'hw7code'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 133a HW7 Code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plotdata        = hw7code.plotdata:main',
            'plotjoints      = hw7code.plotjoints:main',
            'plottranslation = hw7code.plottranslation:main',
            'plotorientation = hw7code.plotorientation:main',
            'plotcondition   = hw7code.plotcondition:main',
            'hw7p1           = hw7code.hw7p1:main',
            'hw7p2           = hw7code.hw7p2:main',
            'hw7p3           = hw7code.hw7p3:main',
        ],
    },
)
