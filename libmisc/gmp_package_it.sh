#! /bin/bash

TMPDIR=/tmp

suffix=${1:-}

package_name=libgmp-dev-mersenne-trinomial${suffix}

dir=$TMPDIR/$(mktemp -d dir-XXXXXX) || exit 1

mkdir -p $dir/DEBIAN

cat << EOF > $dir/DEBIAN/control
Package: $package_name
Version: 6.3.0
Architecture: amd64
Maintainer: Not maintained, specialized one-time purpose
Description: Custom-compiled version of files installed by libgmp-dev
EOF


mkdir -p $dir/usr/local/lib/pkgconfig

cd       $dir/usr/local/lib

cp -p        /usr/local/lib/libgmp.a .
cp -p        /usr/local/lib/libgmp.la .
cp -p        /usr/local/lib/libgmp.so.10.5.0 .
ln -s                       libgmp.so.10.5.0 ./libgmp.so.10
ln -s                       libgmp.so.10.5.0 ./libgmp.so

cd                          pkgconfig
cp -p        /usr/local/lib/pkgconfig/gmp.pc .


mkdir -p $dir/usr/local/include

cd       $dir/usr/local/include

cp -p        /usr/local/include/gmp.h .



mkdir -p $dir/usr/local/share/info

cd       $dir/usr/local/share/info

cp -p        /usr/local/share/info/gmp.info .
cp -p        /usr/local/share/info/gmp.info-1 .
cp -p        /usr/local/share/info/gmp.info-2 .


cd $TMPDIR
dpkg-deb --root-owner-group --build $dir
