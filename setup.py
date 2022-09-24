from setuptools import setup

setup(name='dmp',
      version='0.1',
      description='Distributional Dynamic Movement Primitives',
      url='https://github.com/pritoms/Distributional-Dynamic-Movement-Primitives-D-DMPs',
      author='Pritom Sarker',
      author_email='pritoms@gmail.com',
      license='MIT',
      packages=['dmp'],
      install_requires=[
          'numpy',
          'matplotlib',
          'torch'
      ],
      zip_safe=False)
