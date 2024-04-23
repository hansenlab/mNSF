from distutils.core import setup
setup(
  name = 'mNSF',       
  packages = ['mNSF'],   
  version = '0.1',     
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'multi-sample non-negative spatial factorization',   # Give a short description about your library
  author = "Yi Wang, Kyla Woyshner, Chaichontat Sriworarat, Loyal Goff, Genevieve Stein-O'Brien, Kasper D. Hansen",                   
  author_email = 'yiwangthu4@gmail.com',      
  url = 'https://github.com/hansenlab/mNSF/',   
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['spatial', 'factorization', 'multi-sample'],   
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package

    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',

    'License :: OSI Approved :: MIT License',   # Again, pick a license

    'Programming Language :: Python :: 3.10',      

  ],
)
