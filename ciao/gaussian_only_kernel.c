/*                                                                
**  Copyright (C) 2004-2008  Smithsonian Astrophysical Observatory 
*/                                                                

/*                                                                          */
/*  This program is free software; you can redistribute it and/or modify    */
/*  it under the terms of the GNU General Public License as published by    */
/*  the Free Software Foundation; either version 2 of the License, or       */
/*  (at your option) any later version.                                     */
/*                                                                          */
/*  This program is distributed in the hope that it will be useful,         */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           */
/*  GNU General Public License for more details.                            */
/*                                                                          */
/*  You should have received a copy of the GNU General Public License along */
/*  with this program; if not, write to the Free Software Foundation, Inc., */
/*  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.             */
/*                                                                          */

#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <dsnan.h>
#include <dslib.h>

#define SUBPIX  10
extern short evaluate_kernels( char *wever, 
			       float minrad,
			       float maxrad,
			       float radstep,
			       char *radscale,
			       float ***kernels,
			       long **kx,
			       long **ky,
			       long **dkx,
			       long **dky,
			       long **nkvals,
			       float **scales
			       );


short evaluate_kernels( char *wever, 
			float minrad,
			float maxrad,
			float numrad,
			char *linlog,
			float ***kernels,
			long **kx,
			long **ky,
			long **dkx,
			long **dky,
			long **nkvals,
			float **scales
			)
{

  short nks;
  long ii,jj,kk;

  float **_kernels;
  long *_kx;
  long *_ky;
  long *_dkx;
  long *_dky;
  long *_nkvals;
  float radfac;

  float *_scale;
 
  double lmin;
  double lmax;
  double lstep;

  double subpix = SUBPIX+1;
  double subpix_area = 1.0/(subpix*subpix);
  double subpix_step = 1.0/(SUBPIX);

  if ( strcmp(linlog, "log" ) == 0 ) {
    lmin = log10(minrad);
    lmax = log10(maxrad);
  } else {
    lmin = minrad;
    lmax = maxrad;
  }
  
  if ( numrad > 1 ) {
    lstep = (lmax - lmin ) / (numrad-1);
  } else {
    lstep = 0;
  }
  
  nks = numrad ; 
  
  *kernels = (float**)calloc(nks,sizeof(float*));
  *kx = (long*)calloc(nks,sizeof(long));
  *ky = (long*)calloc(nks,sizeof(long));
  *dkx = (long*)calloc(nks,sizeof(long));
  *dky = (long*)calloc(nks,sizeof(long));
  *nkvals = (long*)calloc(nks,sizeof(long));
  
  _kernels = *kernels;
  _kx = *kx;
  _ky = *ky;
  _dkx = *dkx;
  _dky = *dky;
  _nkvals = *nkvals;

  radfac = 4; /* go out to 5 sigma */


  *scales = (float*)calloc(nks,sizeof(float));

  _scale = *scales; /* makes pointer math easier */

  for( kk=nks; kk--; ) {
    float rad;
    float rad_sq;
    long pix;
    double max;
 
    if ( strcmp(linlog, "log") == 0 ) {
      rad = pow(10,(lmin + (kk*lstep )));
    } else {
      rad = lmin + (kk*lstep);
    }

    if ( rad >= SUBPIX ) {
      subpix = 1;
      subpix_area = 1;
      subpix_step = 1;
    } else {
      subpix = SUBPIX+1;
      subpix_area = 1.0/(subpix*subpix);
      subpix_step = 1.0/(SUBPIX);
    }

    
    _scale[kk] = rad;

    rad_sq = rad * rad;
    

    _kx[kk] = ((short)(2*radfac*rad)+0.5)+1;
    _ky[kk] = ((short)(2*radfac*rad)+0.5)+1;
    _dkx[kk] = radfac*rad;
    _dky[kk] = radfac*rad;
    
    _nkvals[kk] = (_kx[kk]*_ky[kk]);
    
    _kernels[kk]=(float*)calloc( _nkvals[kk],sizeof(float));
    
    pix=(_nkvals[kk]);
    pix--;


    for (jj=_ky[kk];jj--;) {
      double _jj;


      for (ii=_ky[kk];ii--; ) {
	double _ii;
	float *lker = _kernels[kk];
	float dist;
	float dist_sq;

	short subpix_ii;
	short subpix_jj;
	
	lker[pix]=0;
	
	for ( subpix_jj=subpix;subpix_jj--; ) {
	  _jj = jj - _dky[kk] -0.5+(subpix_jj*subpix_step) ;
	  
	  for ( subpix_ii=subpix;subpix_ii--; ) {
	    _ii = ii - _dkx[kk] -0.5+(subpix_ii*subpix_step) ;

	    dist = sqrt((_jj*_jj)+(_ii*_ii));
	    dist_sq = ( dist * dist);
	    
	    if (*wever == 'g' ) { /* gaussian */
	      if ( 0 == rad )
		lker[pix] += 1.0*subpix_area;
	      else
		lker[pix] += exp( -0.5 * dist_sq / rad_sq)*subpix_area;
	    } 
	      
	      if ( mm < rad )
		lker[pix] += subpix_area*(1-(mm/rad));
	      else 
		lker[pix] += 0;

	    } else {
	      nks = 0;
	      return(nks);
	    }

	  } /* end subpix_ii */
	} /* end subpix_jj */


	pix--;
      } /* end for ii */
    } /* end for jj */

    /* Normalize kernel to max=1 */
    pix=(_nkvals[kk]);
    pix--;
    max=0;
    for (jj=_ky[kk];jj--;) {
      for (ii=_ky[kk];ii--; ) {
	float *lker = _kernels[kk];
	if ( lker[pix]>max ) max= lker[pix];
	pix--;
      }
    }
    
    pix=(_nkvals[kk]);
    pix--;
    for (jj=_ky[kk];jj--;) {
      for (ii=_ky[kk];ii--; ) {
	float *lker = _kernels[kk];
	lker[pix] /= max;
	pix--;
      }
    }



  } /* end loop over kk */




  return(nks);
}



int sortLo2Hi(
                     const void *ii,         /* void pointer to float */
                     const void *jj          /* void pointer to float */
                     );
int sortLo2Hi(
                     const void *ii,         /* void pointer to float */
                     const void *jj          /* void pointer to float */
                     )
{
  double *xx, *yy;
  int    retval=0;
  
  xx = (double *) ii;              /* cast voids to floats */
  yy = (double *) jj;
  
  if ( *xx > *yy ) 
    {
      retval = 1;         /* xx is bigger than yy  */
    }
  else if ( *xx < *yy )
    {
      retval = -1;        /* xx is smaller than yy */
    }
  else
    retval = 0;


  return(retval);
}



short get_inradkernels( char *wever, 
			double *inrad,
			long *lAxes,
			float ***kernels,
			long **kx,
			long **ky,
			long **dkx,
			long **dky,
			long **nkvals,
			float **scales
			);


short get_inradkernels( char *wever, 
			double *inrad,
			long *lAxes,
			float ***kernels,
			long **kx,
			long **ky,
			long **dkx,
			long **dky,
			long **nkvals,
			float **scales
			)
{
  
  long npix;
  long nks;
  long nnk;
  long ii,kk,jj;
  double lastval;
  double *loc_val;
  float **_kernels;
  long *_kx;
  long *_ky;
  long *_dkx;
  long *_dky;
  long *_nkvals;
  float radfac;
  float *_scale;

  double subpix = SUBPIX+1;
  double subpix_area = 1.0/(subpix*subpix);
  double subpix_step = 1.0/(SUBPIX);

  double max_non_null=-DBL_MAX;

  dsErrList *err_p;
  
  dsErrCreateList( &err_p);
  
  npix = lAxes[0]*lAxes[1];

  for (ii=npix;ii--; ) {
    if ( inrad[ii] > max_non_null )
      max_non_null = inrad[ii];
  }

  for (ii=npix;ii--; ) {
    if ( ds_dNAN( inrad[ii] ) ) {
      dsErrAdd( err_p, dsBOUNDSERR, Accumulation, Custom, 
		"WARNING: NaN or out-of-subspace pixel found, replacing radius with a value=%g\n", max_non_null );

      inrad[ii] = max_non_null;
    }
    
  }

  dsErrPrintList( err_p, dsErrTrue );

  
  loc_val = (double*)calloc( npix, sizeof(double));
  memcpy( loc_val, inrad, npix*sizeof(double));

  qsort( loc_val,npix , sizeof(double), sortLo2Hi);

  lastval = loc_val[npix-1]+1;
  nks=0;
  for (ii=npix;ii--;) {
    if (loc_val[ii]!=lastval) {
      nks+=1;
      lastval = loc_val[ii];
    }
  }
  
  
  *scales = (float*)calloc(nks,sizeof(float));
  _scale = *scales; /* makes pointer math easier */
  
  nnk = 0;
  lastval = loc_val[0]-1;
  for(ii=0;ii<npix;ii++) {
    if ( loc_val[ii] != lastval ) {
      _scale[nnk] = loc_val[ii];
      nnk++;
      lastval = loc_val[ii];
    }
  }
  
  free( loc_val );
  
  /* Okay, now cut-n-paste from above */
  
  *kernels = (float**)calloc(nks,sizeof(float*));
  *kx = (long*)calloc(nks,sizeof(long));
  *ky = (long*)calloc(nks,sizeof(long));
  *dkx = (long*)calloc(nks,sizeof(long));
  *dky = (long*)calloc(nks,sizeof(long));
  *nkvals = (long*)calloc(nks,sizeof(long));
  
  _kernels = *kernels;
  _kx = *kx;
  _ky = *ky;
  _dkx = *dkx;
  _dky = *dky;
  _nkvals = *nkvals;

  radfac = 4; /* go out to 5 sigma */



  for( kk=nks; kk--; ) {
    float rad;
    float rad_sq;
    long pix;
    double max;

    rad = _scale[kk];

    if ( rad >= SUBPIX ) {
      subpix = 1;
      subpix_area = 1;
      subpix_step = 1;
    } else {
      subpix = SUBPIX+1;
      subpix_area = 1.0/(subpix*subpix);
      subpix_step = 1.0/(SUBPIX);
    }

    
    _scale[kk] = rad;

    rad_sq = rad * rad;
    

    _kx[kk] = ((short)(2*radfac*rad)+0.5)+1;
    _ky[kk] = ((short)(2*radfac*rad)+0.5)+1;
    _dkx[kk] = radfac*rad;
    _dky[kk] = radfac*rad;
    
    _nkvals[kk] = (_kx[kk]*_ky[kk]);
    
    _kernels[kk]=(float*)calloc( _nkvals[kk],sizeof(float));
    
    pix=(_nkvals[kk]);
    pix--;


    for (jj=_ky[kk];jj--;) {
      double _jj;


      for (ii=_ky[kk];ii--; ) {
	double _ii;
	float *lker = _kernels[kk];
	float dist;
	float dist_sq;

	short subpix_ii;
	short subpix_jj;
	
	lker[pix]=0;
	
	for ( subpix_jj=subpix;subpix_jj--; ) {
	  _jj = jj - _dky[kk] -0.5+(subpix_jj*subpix_step) ;
	  
	  for ( subpix_ii=subpix;subpix_ii--; ) {
	    _ii = ii - _dkx[kk] -0.5+(subpix_ii*subpix_step) ;

	    dist = sqrt((_jj*_jj)+(_ii*_ii));
	    dist_sq = ( dist * dist);
	    
	    if (*wever == 'g' ) { /* gaussian */
	      if ( 0 == rad )
		lker[pix] += 1.0*subpix_area;
	      else
		lker[pix] += exp( -0.5 * dist_sq / rad_sq)*subpix_area;
	    } 
	      
	      if ( mm < rad )
		lker[pix] += subpix_area*(1-(mm/rad));
	      else 
		lker[pix] += 0;

	    } else {
	      nks = 0;
	      return(nks);
	    }

	  } /* end subpix_ii */
	} /* end subpix_jj */


	pix--;
      } /* end for ii */
    } /* end for jj */

    /* Normalize kernel to max=1 */
    pix=(_nkvals[kk]);
    pix--;
    max=0;
    for (jj=_ky[kk];jj--;) {
      for (ii=_ky[kk];ii--; ) {
	float *lker = _kernels[kk];
	if ( lker[pix]>max ) max= lker[pix];
	pix--;
      }
    }
    
    pix=(_nkvals[kk]);
    pix--;
    for (jj=_ky[kk];jj--;) {
      for (ii=_ky[kk];ii--; ) {
	float *lker = _kernels[kk];
	lker[pix] /= max;
	pix--;
      }
    }



  } /* end loop over kk */





  return(nks);


}
