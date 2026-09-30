# Project Directory Structure

The directory hierarchy is described as follows:

```text
├── aicpu_common                                        # Common function implementation for AICPU
├── cmake                                               # Project build directory
│   ├── third_party                                     # Build configuration directory for third-party dependencies
│   ├── aarch64-hcc-toolchain.cmake                     # Toolchain configuration file for project build
│   ├── config.cmake                                    # Build option configuration file for the project
│   ├── makeself.cmake                                  # Custom makeself packaging file for the project
│   ├── package.cmake                                   # Build, packaging, and installation configuration file for the project
│   ├── Third_Party_Open_Source_Software_List.yaml      # List of third-party software libraries used by the project
│   └── variables.cmake                                 # Build parameter configuration file for the project
├── docs                                                # Project documentation directory (zh for Chinese, en for English)
├── include                                             # Common header file directory of the project
│   └── nnopbase                                        # Header files of the nnopbase module
│        ├── aclnn                                      # Header files required by ACLNN interfaces
│        └── opdev                                      # Header files required by operator development
│            ├── aicpu                                  # Header files related to AICPU operator development
│            └── op_common                              # Common operator interface header files
├── pkg_inc                                             # Inter-package interface header file directory of the project
│   └── op_common                                       # Header files of the op_common module
│       ├── atvoss                                      # ATVOSS interface header files, including broadcast, elewise, etc.
│       ├── log                                         # Log-related interface header files
│       ├── op_host                                     # Host-side interface header files
│       ├── aicpu_common                                # Common AICPU function header files
│       └── op_kernel                                   # Kernel-side interface header files
├── scripts                                             # Directory for project script files
├── src
|   └── nnopbase                                        # Source code directory of nnopbase
│       ├── aicpu                                       # AICPU framework code
│       ├── common                                      # Common files of nnopbase
│       ├── composite_op                                # Composite multi-operator framework code
│       ├── individual_op                               # Single-operator framework code
│       ├── stub                                        # Packaging scripts for cross-compilation scenarios
│       ├── tls_guardian                                # Patches for resolving glibc issues
│       └── CMakeLists.txt                              # Build configuration file of the nnopbase module
│   └── op_common                                       # Source code implementation of op_common
│       ├── atvoss                                      # Source code implementation of ATVOSS interfaces
│       ├── log                                         # Source code implementation of log interfaces
│       └── op_host                                     # Source code implementation of host-side interfaces
├── tests                                               # Test project directory
│   ├── CMakeLists.txt
│   └── ut                                              # UT case project
│       ├── CMakeLists.txt                              # CMake script of the UT project
│       └── op_common                                   # Test project of op_common
├── build.sh                                            # Project build script
├── CMakeLists.txt                                      # Entry CMakeLists of the project
├── CONTRIBUTING.md                                     # Contribution guide of the project
├── install_deps.sh                                     # Script for installing project dependencies
├── LICENSE                                             # Open source license information of the project
├── OAT.xml                                             # Configuration script used by the repository tooling to check license compliance
├── README.md                                           # Overall introduction document of the project
├── SECURITY.md                                         # Security statement of the project
└── version.info                                        # Version information of the project
```
