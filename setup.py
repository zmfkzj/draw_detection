import setuptools
from subprocess import check_call

#draw detection tool

install_requires = \
    [
        'pillow',
        'numpy',
        'chardet'
    ]

check_call("conda install -y --file requirements.txt".split())
setuptools.setup(
    name="ddt",
    version="0.0.7",
    author="zmfkzj",
    author_email="qlwlal@naver.com",
    description="이미지에 detection 결과를 그리기 위한 모듈",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    include_package_data = True,
    data_files = [('ddt',['ddt/NanumGothicBold.ttf'])]
)
