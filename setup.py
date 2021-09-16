import setuptools
from subprocess import check_call
import os.path as osp
from pathlib import Path

#draw detection tool

install_requires = \
    [
        'pillow',
        'numpy',
        'chardet'
    ]

root = Path(__file__).parent
check_call("conda install -y --file".split()+[str(root/"requirements.txt")])
setuptools.setup(
    name="ddt",
    version="0.0.11",
    author="zmfkzj",
    author_email="qlwlal@naver.com",
    description="이미지에 detection 결과를 그리기 위한 모듈",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    include_package_data = True,
    data_files = [('ddt',['ddt/NanumGothicBold.ttf'])]
)
