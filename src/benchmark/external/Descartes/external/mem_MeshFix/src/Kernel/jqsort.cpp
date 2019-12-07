/****************************************************************************
* TMesh                                                                  *
*                                                                           *
* Consiglio Nazionale delle Ricerche                                        *
* Istituto di Matematica Applicata e Tecnologie Informatiche                *
* Sezione di Genova                                                         *
* IMATI-GE / CNR                                                            *
*                                                                           *
* Authors: Marco Attene                                                     *
* Copyright(C) 2012: IMATI-GE / CNR                                         *
* All rights reserved.                                                      *
*                                                                           *
* This program is dual-licensed as follows:                                 *
*                                                                           *
* (1) You may use TMesh as free software; you can redistribute it and/or *
* modify it under the terms of the GNU General Public License as published  *
* by the Free Software Foundation; either version 3 of the License, or      *
* (at your option) any later version.                                       *
* In this case the program is distributed in the hope that it will be       *
* useful, but WITHOUT ANY WARRANTY; without even the implied warranty of    *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
* (2) You may use TMesh as part of a commercial software. In this case a *
* proper agreement must be reached with the Authors and with IMATI-GE/CNR   *
* based on a proper licensing contract.                                     *
*                                                                           *
****************************************************************************/

#define USE_STD_SORT

#ifdef USE_STD_SORT
#include <vector>
#include <algorithm>
#endif

namespace T_MESH
{

#ifndef USE_STD_SORT
inline void jswap(void *v[], int i, int j)
{
 void *temp = v[i];
 v[i] = v[j];
 v[j] = temp;
}

void jqsort_prv(void *v[], int left, int right, int (*comp)(const void *, const void *))
{
 register int i, last;
 
 if (left >= right) return;
 jswap(v, left, (left+right)/2);
 last = left;
 for (i = left+1; i <= right; i++)
  if ((*comp)(v[i], v[left]) < 0) jswap(v, ++last, i);
 jswap(v, left, last);
 jqsort_prv(v, left, last-1, comp);
 jqsort_prv(v, last+1, right, comp);
}

void jqsort(void *v[], int numels, int(*comp)(const void *, const void *))
{
	jqsort_prv(v, 0, numels-1, comp);
}
#else

class compobj
{
	int(*comp)(const void *, const void *);

public:
	compobj(int(*c)(const void *, const void *)) { comp = c; }

	bool operator()(void *a, void *b) {	return (comp(a, b) < 0); }
};

void jqsort(void *v[], int numels, int(*comp)(const void *, const void *))
{
	compobj a(comp);
	std::sort(v, v + numels, a);
}
#endif

} //namespace T_MESH

