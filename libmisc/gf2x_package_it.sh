#! /bin/bash

TMPDIR=/tmp

suffix=${1:-}

package_name=libgf2x-dev-mersenne-trinomial${suffix}

dir=$TMPDIR/$(mktemp -d dir-XXXXXX) || exit 1

mkdir -p $dir/DEBIAN

cat << EOF > $dir/DEBIAN/control
Package: $package_name
Version: 1.3.99
Architecture: amd64
Maintainer: Not maintained, specialized one-time purpose
Description: Custom-compiled version of files installed by libgf2x-dev
EOF


mkdir -p $dir/usr/local/lib/pkgconfig

cd       $dir/usr/local/lib

cp -p        /usr/local/lib/libgf2x.a .
cp -p        /usr/local/lib/libgf2x.la .
cp -p        /usr/local/lib/libgf2x.so.4.0.0 .
ln -s                       libgf2x.so.4.0.0 ./libgf2x.so.4
ln -s                       libgf2x.so.4.0.0 ./libgf2x.so

cd                          pkgconfig
cp -p        /usr/local/lib/pkgconfig/gf2x.pc .


mkdir -p $dir/usr/local/include/gf2x

cd       $dir/usr/local/include

cp -p        /usr/local/include/gf2x.h .

cd gf2x
cp -p        /usr/local/include/gf2x/gf2x-*.h .


mkdir -p $dir/usr/local/share/info

cd       $dir/usr/local/share/info

cp -p        /usr/local/share/info/gf2x.info .


cd $TMPDIR
dpkg-deb --root-owner-group --build $dir
