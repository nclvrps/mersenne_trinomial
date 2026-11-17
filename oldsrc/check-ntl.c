/* Checks an extended log file with NTL.
   Usage: check-ntl r < xxx.log-ext
   where xxx.log-ext contains lines 's k pxxx', for example:
   1 2 p7
   meaning that x^r+x^s+1 has a factor of degree k, whose hexadecimal encoding
   is pxxx. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <NTL/GF2X.h>

NTL_CLIENT

/* checks that the polynomial whose hexadecimal representation is s
   divides x^r+x^s+1 */
static void
check (unsigned long r, unsigned long s, unsigned long k, char *str,
       int verbose)
{
  unsigned long n, i, j;
  GF2X f, t, u, v;
  char c;
  
  n = 1 + (k / 4); /* length of str */
  for (i = 0; i < n; i++)
    {
      c = str[i]; /* '0' to 'f' */
      c = (c < 'a') ? c - '0' : c - 'a' + 10;
      for (j = 4; j-- > 0;)
        if (c & (1 << j))
          SetCoeff (f, (n - 1 - i) * 4 + j);
    }
  if (deg (f) != k)
    {
      fprintf (stderr, "Error, factor degree (%lu) does not match given degree (%lu)\n", deg (f), k);
      printf ("s=%lu k=%lu\n", s, k);
      cout << f << endl;
      exit (1);
    }

  GF2XModulus F(f);

  SetX (t);
  PowerMod (t, t, r, F);
  SetX (u);
  PowerMod (u, u, s, F);
  add (t, t, u);
  SetCoeff (v, 0); /* v = 1 */
  add (t, t, v);
  if (IsZero (t) == 0)
    {
      fprintf (stderr, "Error, given factor of degree %lu does not divide x^%lu+x^%lu+1\n", k, r, s);
      exit (1);
    }
  if (verbose)
    printf ("%lu %lu\n", s, k);
}

int
main (int argc, char *argv[])
{
  unsigned long r, s, k;
  int c;
  char *factor = NULL;
  unsigned long alloc = 0; /* allocated size of factor */
  unsigned long n; /* length of factor in hexadecimal */
  unsigned long trinomial = 0;
  unsigned long primitive = 0;
  int verbose = 0;
  char str[10];

  if (argc >= 2 && strcmp (argv[1], "-v") == 0)
    {
      verbose = 1;
      argc --;
      argv ++;
    }

  r = atoi (argv[1]);
  
  while (!feof (stdin))
    {
      if (scanf ("%lu", &s) != 1)
        {
          if (feof (stdin))
            break;
          fprintf (stderr, "Error while reading s at line %lu\n",
                   trinomial + 1);
          exit (1);
        }
      trinomial ++;
      k = 0;
      if (scanf ("%lu", &k) != 1 || k == r)
        {
          if (scanf ("%s", str) != 1 || (strcmp (str, "primitive") != 0 &&
                                         strcmp (str, "u") != 0))
            {
              fprintf (stderr, "Error while reading k and/or primitive\n");
              exit (1);
            }
          /* trinomial is claimed primitive */
          primitive ++;
          continue;
        }

      n = 1 + (k / 4); /* a polynomial of degree k has k+1 coefficients,
                          and ceil((k+1)/4) = 1 + floor(k/4) */
      if (alloc < n + 1)
        {
          factor = (char*) realloc (factor, (n + 1) * sizeof (char));
          alloc = n;
        }
      while (getchar () != 'p');
      if (scanf ("%s", factor) != 1)
        {
          fprintf (stderr, "Error while reading factor\n");
          exit (1);
        }
      factor[n] = '\0';
      check (r, s, k, factor, verbose);
    }
  fprintf (stderr, "Read %lu trinomials (%lu primitive or unknown)\n",
           trinomial, primitive);
  free (factor);
  return 0;
}

