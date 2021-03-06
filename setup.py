import setuptools
from subprocess import check_call
import os.path as osp
from pathlib import Path

#draw detection tool

install_requires = \
    [
        'pillow',
        'numpy',
        'chardet',
        'opencv-python'
    ]

setuptools.setup(
    name="ddt",
    version="0.0.18",
    author="zmfkzj",
    author_email="qlwlal@naver.com",
    description="이미지에 detection 결과를 그리기 위한 모듈",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    include_package_data = True,
    data_files = [('ddt',['ddt/NanumGothicBold.ttf'])]
)
