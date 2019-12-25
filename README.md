<!---
  PyOgmaNeo
  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the PYOGMANEO_LICENSE.md file included in this distribution.
--->

# PyOgmaNeo, V2

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction 

Welcome to the [Ogma](https://ogmacorp.com) PyOgmaNeo2 library, which contains Python bindings to the [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2) library.

Note that there are two libraries implementing SPH: [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2), and an embedded (CPU only) version [EOgmaNeo](https://github.com/ogmacorp/EOgmaNeo).

There is also a [deprecated version](https://github.com/ogmacorp/OgmaNeo) of OgmaNeo2 that contains an outdated implementation of SPH. Please use OgmaNeo2 (which this repository provides Python bindings for) or EOgmaNeo if possible.

## Requirements

An install of [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2) is required before installing the bindings.

Additionally this binding requires an installation of [SWIG](http://www.swig.org/) v3+

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via for example ```sudo apt-get install swig3.0``` (Debian).
- Windows requires installation of SWIG (v3). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).
- Mac OSX can use Homebrew to install the latest SWIG (for example, see .travis/install_swig.sh Bash script).

## Installation

The following example can be used to build the Python package:

> python3 setup.py install --user  

or create a wheel file for installation via pip:

> python3 setup.py bdist_wheel  
> pip3 install dist/*.whl --user  

## Importing and Setup

The PyOgmaNeo2 module can be imported using:

```python
import pyogmaneo
```

Refer to the `examples` directory for further details.

## Documentation

Documentation is [available here](https://ogmacorp.github.io/PyOgmaNeo2-Docs/overview.html).
The sources of the documentation are also included in this repository.

## Contributions

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for information on making contributions to PyOgmaNeo2.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [PYOGMANEO_LICENSE.md](./PYOGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeo Copyright (c) 2016-2019 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
