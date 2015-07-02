# Essential
sudo apt-get install cmake subversion git python-dev openjdk-6-jdk libtool libglib2.0-dev  autopoint autoconf2.13 freeglut3-dev libgtk2.0-dev libjpeg8-dev  libusb-1.0-0-dev libbluetooth-dev libsuitesparse-dev cmake g++ libgsl0-dev libfftw3-dev ant libusb-dev libxml2-dev doxygen libboost-all-dev libncurses5-dev python-scipy build-essential python-numpy python-pandas python-tables python-matplotlib libbullet-dev ipython-notebook  pkg-config libqglviewer-qt4-dev python-scikits-learn python-networkx

# pip with latest scikits learn module
sudo easy_install pip
sudo pip install scikits.learn --upgrade
sudo pip install numpy --upgrade

# VTK
sudo apt-get install libvtk5-dev # for opencv_viz module

# Enhancements
sudo apt-get install htop valgrind kcachegrind libsdl1.2-dev python-qt4-dev

# QGLViewer
sudo add-apt-repository ppa:christophe-pradal/openalea
sudo add-apt-repository ppa:christophe-pradal/vplants
sudo apt-get install pyqglviewer

# Qt5
sudo apt-add-repository ppa:ubuntu-sdk-team/ppa
sudo apt-get install qtdeclarative5-dev
