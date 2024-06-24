from setuptools import find_packages,setup

setup(

    name='mcq_generator',
    version='0.0.7',
    author='shreyas kale',
    author_email='kaleshreyas4@gmail.com',
    install_requires=['gemini','langchain','streamlit','python-dotenv','PyPDF2'],
    packages=find_packages()
)