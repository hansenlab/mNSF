from distutils.core import setup
setup(
  name = 'mNSF',
  packages = ['mNSF'],
  version = '0.1.5',
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'multi-sample non-negative spatial factorization',   #
  author = "Yi Wang, Kyla Woyshner, Chaichontat Sriworarat, Loyal Goff, Genevieve Stein-O'Brien, Kasper D. Hansen",
  author_email = 'yiwangthu4@gmail.com',
  url = 'https://github.com/hansenlab/mNSF/',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
  keywords = ['spatial', 'factorization', 'multi-sample'],
  install_requires=[
          'anndata', 'click', 'dill','matplotlib', 'numpy', 'pandas', 'pip',
          'scanpy', 'squidpy', 'tensorflow==2.13.*', 'tensorflow-probability==0.21.*'],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package

    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',

  ],
)
