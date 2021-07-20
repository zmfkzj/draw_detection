import setuptools

#draw detection tool

install_requires = \
    [
        'opencv-python',
        'pillow',
        'numpy'
    ]

setuptools.setup(
    name="ddt",
    version="0.0.1",
    author="zmfkzj",
    author_email="qlwlal@naver.com",
    description="이미지에 detection 결과를 그리기 위한 모듈",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
)