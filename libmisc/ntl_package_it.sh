#! /bin/bash

TMPDIR=/tmp

suffix=${1:-}

package_name=libntl-dev-mersenne-trinomial${suffix}

dir=$TMPDIR/$(mktemp -d dir-XXXXXX) || exit 1

mkdir -p $dir/DEBIAN

cat << EOF > $dir/DEBIAN/control
Package: $package_name
Version: 11.6.0
Architecture: amd64
Maintainer: Not maintained, specialized one-time purpose
Description: Custom-compiled version of files installed by libntl-dev (but not shared library)
EOF


mkdir -p $dir/usr/local/lib/pkgconfig

cd       $dir/usr/local/lib

cp -p        /usr/local/lib/libntl.a .

cd                          pkgconfig
cp -p        /usr/local/lib/pkgconfig/ntl.pc .


mkdir -p $dir/usr/local/include/NTL

cd       $dir/usr/local/include/NTL

cp -p        /usr/local/include/NTL/*.h .


mkdir -p $dir/usr/local/share/doc/NTL

cd       $dir/usr/local/share/doc/NTL

cp -p        /usr/local/share/doc/NTL/* .


cd $TMPDIR
dpkg-deb --root-owner-group --build $dir
