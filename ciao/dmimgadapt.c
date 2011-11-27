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


#include <dslib.h>
#include <dsnan.h>
#include <histlib.h>
#include <math.h>
#include <float.h>
#include <cxcregion.h>


int dmimgfilter(void);

extern short get_inradkernels( char *wever, 
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

short *get_image_mask( dmBlock *inBlock, void *data, dmDataType dt, 
		       long *lAxes, regRegion *dss, long null, short has_null, 
		       dmDescriptor *xAxis, dmDescriptor *yAxis );


double get_image_value( void *data, dmDataType dt, 
			long xx, long yy, long *lAxes, 
			short *mask );

dmDataType get_image_data( dmBlock *inBlock, void **data, long **lAxes,
			   regRegion **dss, long *nullval, short *nullset );


short  get_image_wcs( dmBlock *imgBlock, dmDescriptor **xAxis, 
		      dmDescriptor **yAxis );




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






short *get_image_mask( dmBlock *inBlock, void *data, dmDataType dt, 
		       long *lAxes, regRegion *dss, long null, short has_null, 
		       dmDescriptor *xAxis, dmDescriptor *yAxis )
{
  long npix = lAxes[0] * lAxes[1];
  short *mask;
  long xx, yy;
  mask = (short*)calloc( npix, sizeof(short));

  
  for ( xx=lAxes[0]; xx--; ) {
    for ( yy=lAxes[1]; yy--; ) {
      double dat;
      long idx;
      idx = xx + ( yy * lAxes[0] );
      
      dat = get_image_value( data, dt, xx, yy, lAxes, NULL );
      
      /* Now ... if it is an integer data type, it could possibly have a
	 null value. Check for that */
      if ( ( has_null && ( dat == null ) ) ||
	   ds_dNAN( dat ) ) {
	continue;
      }
            
      /* If the image has a data sub space (aka a region filter applied)
	 then need to convert coords to physical and check */
      if ( dss && xAxis ) {
	double pos[2];
	double loc[2];
	pos[0]=xx+1;
	pos[1]=yy+1;
	
	if (yAxis) {  /* If no y axis, then xAxis has 2 components */
	  dmCoordCalc_d( xAxis, pos, loc );
	  dmCoordCalc_d( yAxis, pos+1, loc+1 );
	} else {
	  dmCoordCalc_d( xAxis, pos, loc );
	}
	if ( !regInsideRegion( dss, loc[0], loc[1] ) )
	  continue;
      }
      
      mask[idx] = 1;
    }
  }


  return(mask );
}



double get_image_value( void *data, dmDataType dt, 
			long xx, long yy, long *lAxes, 
			short *mask )
{

  long npix = xx + (yy * lAxes[0] );
  double retval;

  /* Okay, first get all the data from the different data types.  
     Cast everything to doubles */

  switch ( dt ) {
    
  case dmBYTE: {
    unsigned char *img = (unsigned char*)data;
    retval = img[npix];
    break;
  }
    
  case dmSHORT: {
    short *img = (short*)data;
    retval = img[npix];
    break;
  }
    
  case dmUSHORT: {
    unsigned short *img = (unsigned short*)data;
    retval = img[npix];
    break;
  }
    
  case dmLONG: {
    long *img = (long*)data;
    retval = img[npix];
    break;
  }
    
  case dmULONG: {
    unsigned long *img = (unsigned long*)data;
    retval = img[npix];
    break;
  }
    
  case dmFLOAT: {
    float *img = (float*)data;
    retval = img[npix];
    break;
  }
  case dmDOUBLE: {
    double *img = (double*)data;
    retval = img[npix];
    break;
  }
  default:
    ds_MAKE_DNAN( retval );

  }


  if ( mask ) {
    if ( !mask[npix] ) {
      ds_MAKE_DNAN( retval );
    }
  }


  return(retval);

}



/* Load the data into memory,  check for DSS, null values */
dmDataType get_image_data( dmBlock *inBlock, void **data, long **lAxes,
			   regRegion **dss, long *nullval, short *nullset )
{

  dmDescriptor *imgDesc;
  dmDataType dt;
  dmDescriptor *grp;
  dmDescriptor *imgdss;

  long naxes;
  long npix;
  char ems[1000];

  *nullval = INDEFL;
  *dss = NULL;
  *nullset = 0;
  
  imgDesc = dmImageGetDataDescriptor( inBlock );

  /* Sanity check, only 2D images */
  naxes = dmGetArrayDimensions( imgDesc, lAxes );
  if ( naxes != 2 ) {
    return( dmUNKNOWNTYPE );
  }
  npix = (*lAxes)[0] * (*lAxes)[1];
  dt = dmGetDataType( imgDesc );


  /* Okay, first lets get the image descriptor */
  grp = dmArrayGetAxisGroup( imgDesc, 1 );
  dmGetName( grp, ems, 1000);
  imgdss = dmSubspaceColOpen( inBlock, ems );
  if ( imgdss )
    *dss = dmSubspaceColGetRegion( imgdss);
  
  
  switch ( dt ) 
    {
    case dmBYTE:
      *data = ( void *)calloc( npix, sizeof(char ));
      dmGetArray_ub( imgDesc, (unsigned char*) *data, npix );
      if ( dmDescriptorGetNull_l( imgDesc, nullval) == 0 ) {
	*nullset=0;
      } else
	*nullset=1;
      break;
      
    case dmSHORT:
      *data = ( void *)calloc( npix, sizeof(short ));
      dmGetArray_s( imgDesc, (short*) *data, npix );
      if ( dmDescriptorGetNull_l( imgDesc, nullval) == 0 ) {
	*nullset=0;
      } else
	*nullset=1;
      break;
      
    case dmUSHORT:
      *data = ( void *)calloc( npix, sizeof(short ));
      dmGetArray_us( imgDesc, (unsigned short*) *data, npix );
      if ( dmDescriptorGetNull_l( imgDesc, nullval) == 0 ) {
	*nullset=0;
      } else
	*nullset=1;
      break;
      
    case dmLONG:
      *data = ( void *)calloc( npix, sizeof(long ));
      dmGetArray_l( imgDesc, (long*) *data, npix );
      if ( dmDescriptorGetNull_l( imgDesc, nullval) == 0 ) {
	*nullset=0;
      } else
	*nullset=1;
      break;
      
    case dmULONG:
      *data = ( void *)calloc( npix, sizeof(long ));
      dmGetArray_ul( imgDesc, (unsigned long*) *data, npix );
      if ( dmDescriptorGetNull_l( imgDesc, nullval) == 0 ) {
	*nullset=0;
      } else
	*nullset=1;
      break;
      
    case dmFLOAT:
      *data = ( void *)calloc( npix, sizeof(float ));
      dmGetArray_f( imgDesc, (float*) *data, npix );
      *nullset = 0;
      break;
      
    case dmDOUBLE:
      *data = ( void *)calloc( npix, sizeof(double ));
      dmGetArray_d( imgDesc, (double*) *data, npix );
      *nullset = 0;
      break;
      
    default:
      return( dmUNKNOWNTYPE );
    }

  return(dt);

}




/* Get the WCS descriptor */
short  get_image_wcs( dmBlock *imgBlock, dmDescriptor **xAxis, 
		      dmDescriptor **yAxis )
{
  

  dmDescriptor *imgData;
  long n_axis_groups;

  imgData = dmImageGetDataDescriptor( imgBlock );
  n_axis_groups = dmArrayGetNoAxisGroups( imgData );
  

  /* This is the usual trick ... can have 1 axis group w/ 
     dimensionality 2 (eg a vector column) or can have
     2 axis groups w/ dimensionaity 1 (eg 2 disjoint columns)*/

  if ( n_axis_groups == 1 ) {
    dmDescriptor *pos = dmArrayGetAxisGroup( imgData, 1 );
    dmDescriptor *xcol;
    long n_components;
    
    n_components = dmGetElementDim( pos );
    if ( n_components != 2 ) {
      err_msg("ERROR: could not find 2D image\n");
      return(-1);
    }
    
    xcol = dmGetCpt( pos, 1 );
    
    *xAxis = pos;
    *yAxis = NULL;
    
  } else if ( n_axis_groups == 2 ) {
    dmDescriptor *xcol;
    dmDescriptor *ycol;
  
    xcol = dmArrayGetAxisGroup( imgData, 1 );
    ycol = dmArrayGetAxisGroup( imgData, 2 );

    *xAxis = xcol;
    *yAxis = ycol;
    
  } else {
    err_msg("Invalid number of axis groups\n");
    *xAxis = NULL;
    *yAxis = NULL;
    return(-1);
  }

  return(0);

}





int dmimgadapt(void)
{

  char infile[DS_SZ_PATHNAME];
  char outfile[DS_SZ_PATHNAME];
  char sumfile[DS_SZ_PATHNAME];
  char normfile[DS_SZ_PATHNAME];
  char radfile[DS_SZ_PATHNAME];
  char func[10];
  char radscale[10];
  char inradfile[DS_SZ_PATHNAME];
  char innormfile[DS_SZ_PATHNAME];
  short clobber;
  short verbose;

  void *data;
  long *lAxes;
  regRegion *dss;
  long null;
  short has_null;

  dmDataType dt;
  dmBlock *inBlock;
  dmDescriptor *xdesc, *ydesc;
  
  dmBlock *outBlock;
  dmDescriptor *outDesc;

  dmBlock *outNorm;
  dmDescriptor *outNormDesc;

  dmBlock *outRad;
  dmDescriptor *outRadDesc;

  dmBlock *outSum;
  dmDescriptor *outSumDesc;

  float **kernel;
  long *kx; 
  long *ky;
  long *dkx; 
  long *dky;
  long *nkvals;
  long nkernels;
  float *scales;

  long xx, yy, ix, iy;

  double *outdata;
  short *nullmask;

  double *out_sum;
  double *out_norm;
  double *out_rad;

  unsigned long outpix;

  double minrad;
  double maxrad;
  double numrad;
  float mincounts;

  short kiter;
  short kdir;

  double *norm_data;

  short use_inrad;
  short use_innrm;
  

  /* Get the parameters */
  clgetstr( "infile", infile, DS_SZ_PATHNAME );
  clgetstr( "outfile", outfile, DS_SZ_PATHNAME );
  clgetstr( "function", func, 10 );
  minrad = clgetd( "minrad" );
  maxrad = clgetd( "maxrad" );
  numrad = clgetd( "numrad");
  clgetstr( "radscale", radscale, 10 );
  mincounts = clgetd( "counts" );
  clgetstr( "inradfile", inradfile, DS_SZ_PATHNAME );
  clgetstr( "innormfile", innormfile, DS_SZ_PATHNAME );
  clgetstr( "sumfile", sumfile, DS_SZ_PATHNAME );
  clgetstr( "normfile", normfile, DS_SZ_PATHNAME );
  clgetstr( "radfile", radfile, DS_SZ_PATHNAME );
  clobber = clgetb( "clobber" );
  verbose = clgeti( "verbose" );



  if ( ds_clobber( outfile, clobber, NULL ) != 0 ) {
    return(-1);
  }
  
  if ( strlen( radfile ) && (ds_strcmp_cis(radfile,"none")) ) {
    if ( ds_clobber( radfile, clobber, NULL ) != 0 ) {
      return(-1);
    }
  }
  
  if ( strlen( sumfile ) && (ds_strcmp_cis(sumfile,"none")) ) {
    if ( ds_clobber( sumfile, clobber, NULL ) != 0 ) {
      return(-1);
    }
  }
  
  if ( strlen( normfile ) && (ds_strcmp_cis(normfile,"none")) ) {
    if ( ds_clobber( normfile, clobber, NULL ) != 0 ) {
      return(-1);
    }
  }

  if ( NULL == ( inBlock = dmImageOpen( infile) ) ) {
    err_msg("ERROR: Cannot open image '%s'\n", infile );
    return(-1);
  }
    
  if ( dmUNKNOWNTYPE == ( dt = get_image_data( inBlock, &data, &lAxes, 
					       &dss, &null, &has_null ))) {
    err_msg("ERROR: Cannot get image data or unknown image data-type for "
	    "file '%s'\n", infile);
    return(-1);
  }

  if ( NULL == ( outBlock = dmImageCreate( outfile, dmDOUBLE,lAxes,2 ))){
    err_msg("ERROR: Cannot create output image '%s'\n", outfile );
    return(-1);
  }
  outDesc = dmImageGetDataDescriptor( outBlock );
  dmBlockCopy( inBlock, outBlock, "HEADER");
  ds_copy_full_header( inBlock, outBlock, "dmimgadapt", 0 );
  dmBlockCopyWCS( inBlock, outBlock);

  
  if ( strlen( radfile ) && (ds_strcmp_cis(radfile,"none")) ) {
    if ( NULL == ( outRad = dmImageCreate( radfile, dmDOUBLE,lAxes,2 ))){
      err_msg("ERROR: Cannot create output image '%s'\n", radfile );
      return(-1);
    }
    outRadDesc = dmImageGetDataDescriptor( outRad );
    dmBlockCopy( inBlock, outRad, "HEADER");
    ds_copy_full_header( inBlock, outRad, "dmimgadapt", 0 );
    dmBlockCopyWCS( inBlock, outRad);
  } else {
    outRad = NULL;
  }

  if ( strlen( normfile ) && (ds_strcmp_cis(normfile,"none")) ) {
    if ( NULL == ( outNorm = dmImageCreate( normfile, dmDOUBLE,lAxes,2 ))){
      err_msg("ERROR: Cannot create output image '%s'\n", normfile );
      return(-1);
    }
    outNormDesc = dmImageGetDataDescriptor( outNorm );
    dmBlockCopy( inBlock, outNorm, "HEADER");
    ds_copy_full_header( inBlock, outNorm, "dmimgadapt", 0 );
    dmBlockCopyWCS( inBlock, outNorm);
  } else {
    outNorm = NULL;
  }

  if ( strlen( sumfile ) && (ds_strcmp_cis(sumfile,"none")) ) {
    if ( NULL == ( outSum = dmImageCreate( sumfile, dmDOUBLE,lAxes,2 ))){
      err_msg("ERROR: Cannot create output image '%s'\n", sumfile );
      return(-1);
    }
    outSumDesc = dmImageGetDataDescriptor( outSum );
    dmBlockCopy( inBlock, outSum, "HEADER");
    ds_copy_full_header( inBlock, outSum, "dmimgadapt", 0 );
    dmBlockCopyWCS( inBlock, outSum);
  } else {
    outSum = NULL;
  }


  
  if ( 0 != get_image_wcs( inBlock, &xdesc, &ydesc ) ) {
    err_msg("ERROR: Cannot load WCS for file '%s'\n", infile );
    return(-1);
  }
    
  if ( NULL == ( nullmask = get_image_mask( inBlock, data, dt, 
					    lAxes, dss, null, has_null, 
					    xdesc, ydesc ))){
    
  }

  dmImageClose( inBlock );

  if ( verbose > 1 ) {
    fprintf(stderr,"Pre-computing convolution kernels\n");
  }

  outdata  = (double*)calloc( lAxes[0]*lAxes[1], sizeof(double));
  out_norm = (double*)calloc( lAxes[0]*lAxes[1], sizeof(double));
  out_rad  = (double*)calloc( lAxes[0]*lAxes[1], sizeof(double));
  out_sum  = (double*)calloc( lAxes[0]*lAxes[1], sizeof(double));
  

  
  /* If supplied own scale map, go ahead and get the data now,
     cast data to double */

  if ( ( strlen( inradfile ) > 0 ) &&
       ( ds_strcmp_cis( inradfile, "none") != 0 )) {
    
    dmBlock *irB;
    dmDescriptor *irD;
    long *llAxes;
    
    if ( NULL == ( irB = dmImageOpen( inradfile ))) {
      err_msg("ERROR: Cannot open inradfile='%s'\n", inradfile);
      return(-1);
    }
    if ( NULL == ( irD = dmImageGetDataDescriptor( irB ))) {
      err_msg("ERROR: Cannot open inradfile='%s'\n", inradfile);
      return(-1);
    }
    if ( 2 != dmGetArrayDimensions( irD, &llAxes )) {
      err_msg("ERROR: Only 2D images are supported\n");
      return(-1);
    }
    if ((llAxes[0]!=lAxes[0]) || (llAxes[1]!=lAxes[1])) {
      err_msg("ERROR: inradfile axes don't matchin infile axes\n");
      return(-1);
    }
    
    dmGetArray_d( irD, out_rad, lAxes[0]*lAxes[1]);

    /* Load the scales w/ values from the inradfile */
    nkernels = get_inradkernels( func, out_rad, lAxes, 
				 &kernel, &kx, &ky, &dkx, &dky, &nkvals,
				 &scales);
    
    use_inrad = 1;
    dmImageClose( irB );

    /* If supplied own scale map, go ahead and get the data now,
       cast data to double ;  should only do this when 'use_inrad==1'*/
    
    if ( ( strlen( innormfile ) > 0 ) &&
	 ( ds_strcmp_cis( innormfile, "none") != 0 )) {
      
      dmBlock *irB;
      dmDescriptor *irD;
      long *llAxes;
      
      if ( NULL == ( irB = dmImageOpen( innormfile ))) {
	err_msg("ERROR: Cannot open innormfile='%s'\n", innormfile);
	return(-1);
      }
      if ( NULL == ( irD = dmImageGetDataDescriptor( irB ))) {
	err_msg("ERROR: Cannot open innormfile='%s'\n", innormfile);
	return(-1);
      }
      if ( 2 != dmGetArrayDimensions( irD, &llAxes )) {
	err_msg("ERROR: Only 2D images are supported\n");
	return(-1);
      }
      if ((llAxes[0]!=lAxes[0]) || (llAxes[1]!=lAxes[1])) {
	err_msg("ERROR: innormfile axes don't matchin infile axes\n");
	return(-1);
      }
      
      dmGetArray_d( irD, out_norm, lAxes[0]*lAxes[1]);
      dmImageClose( irB );
      use_innrm = 1;
      
    } else {
      use_innrm = 0;
    }
  
  } else { /* end if inradfile */
    nkernels = evaluate_kernels( func, minrad, maxrad, numrad, radscale,
				 &kernel, &kx, &ky, &dkx, &dky, &nkvals,
				 &scales);
    use_inrad = 0;
    use_innrm = 0;

    if ( ( strlen( innormfile ) > 0 ) &&
	 ( ds_strcmp_cis( innormfile, "none") != 0 )) {
      err_msg("WARNING: Ignoring innormfile. innormfile='%s' is only applicable when inradfile='%s' is supplied.\n", innormfile, inradfile );
    }

  } /* end else not inradfile */
  
  if ( 0 == nkernels ) {
    err_msg("ERROR: Cannot evaluate kernels\n");
    return(-1);
  }






  /* Instead of computing  xx + yy*lAxes[0], just keep a running
     counter of the current pixel number, going backwards thru array */
  outpix=( lAxes[0] * lAxes[1]) -1;
  
  kiter = 0; /* start */
  kdir = 1; /* Positive direction */
  
  /* Okay, first loop over yy and xx which are the image axes */
  if ( verbose > 1 ) {
    fprintf(stderr,"First iteration: determine scales and normalization\n");
  }

  for (yy=lAxes[1];yy--; ) {
    long delta_yy;

    if ( verbose > 2 ) {
      float perc;
      perc = lAxes[1] - yy - 1;
      perc /= lAxes[1];
      perc *= 100;
      fprintf(stderr, "Percent complete: %5.2f%%\r", perc);
      fflush(stderr);
    }

    for (xx=lAxes[0];xx--;) {
      long pix;
      long delta_xx;

      double sum;
      double kernel_sum;
      short kfirst;
      short kstate;
      
      double nan_check;

      double last_sum;
      double last_kernel_sum;

      nan_check = get_image_value( data, dt, xx, yy, lAxes, nullmask );
      if ( ds_dNAN(nan_check) ) {
	out_rad[outpix] = nan_check;
	out_sum[outpix] = nan_check;
	out_norm[outpix] = nan_check;
	outdata[outpix] = nan_check;
	goto end_loop;
      }


      if ( use_inrad ) {
	long wk;
	for ( wk = nkernels; wk--; ) {
	  if ( fabs(out_rad[outpix] - scales[wk]) < 2*FLT_EPSILON*scales[wk] )
	    break;
	}
	kiter = wk;
	
	kfirst = 0;
	kstate = 1;
      } else {
	kfirst = 1;
	kstate = 0;
      }

      
      /* Now loop over the kernels */
      do {
	float *_kernel;  /* These are all just make fewer []'s */
	long _kx;
	long _ky;
	long _dkx;
	long _dky;
	long _nkvals;
	
	_kernel = kernel[kiter];
	_kx = kx[kiter];
	_ky = ky[kiter];
	_dkx = dkx[kiter];
	_dky = dky[kiter];
	_nkvals = nkvals[kiter];
	
	delta_yy = yy - _dky;
	delta_xx = xx - _dkx;
	
	sum = 0;
	kernel_sum = 0;

	pix=_nkvals; /* Same game as w/ the output pixel */
	pix--;

	/* These are looping over the pixels in the kernel */
	for (iy=_ky;iy--;) {
	  long ay;
	  ay = delta_yy + iy;
	  if ( (ay < 0) ||( ay >= lAxes[1] )) {
	    pix-=_kx; /* skip an entire kx row */
	    continue;
	  }
	  
	  for (ix=_kx;ix--;) {
	    long ax;
	    double dater;
	    ax = delta_xx + ix;
	    if (( ax < 0 ) || ( ax >= lAxes[0] )) {
	      pix--; /* skip just one pixel */
	      continue;
	    }
	    
	    dater = get_image_value( data, dt, ax, ay, lAxes, nullmask );
	    if ( ds_dNAN(dater) ) {
	      pix--;
	      continue;
	    }
	    
	    sum += ( dater * _kernel[pix] );
	    kernel_sum += _kernel[pix];
	    pix--;
	    
	  } /* end for ix */
	} /* end for iy */

	/* Alright,  now, if this is the first time we're smoothing
	   this pixel, then we've already started w/ the last
	   kernel size.

	   The idea here is that we start at that kernel.  If after
	   the first smoothing the sum is too small, then we go 
	   in the positive direction.  However, if the sum is too
	   big, we go backwards.
	*/

	if ( kfirst == 1 ) {
	  if ( sum >=  mincounts ) {
	    kdir = -1;
	    kiter--;
	  } else {
	    kdir = 1;
	    kiter++;
	  }
	  kfirst = 0;
	  
	} else {
	  /*
	    now we're stepping through the kernels. 
	  */
	  if ( kstate == 0 ) {
	    if ( -1 == kdir ) {
	      if ( sum >= mincounts ) {
		kiter--;
	      } else {
		kiter++; /* we've gone too low, go up one */
		kstate=1;
		if ( kiter < nkernels ) {
		  sum = last_sum;
		  kernel_sum = last_kernel_sum;
		  _kernel = kernel[kiter];
		  _kx = kx[kiter];
		  _ky = ky[kiter];
		  _dkx = dkx[kiter];
		  _dky = dky[kiter];
		  _nkvals = nkvals[kiter];
		  delta_yy = yy - _dky;
		  delta_xx = xx - _dkx;
		}
	      }
	    } else { 
	      if ( sum >= mincounts ) {
		kstate = 1;
	      } else {
		kiter++;
	      }
	    }
	  } /* end if kstate == 0 */
	  
	} /* end else kfirst == 0 */

	last_kernel_sum = kernel_sum;
	last_sum = sum;
	
	/* Now check the kernel scale */
	if ( kiter < 0 ) {
	  kiter = 0;
	  kstate = 1;
	} else if ( kiter >= nkernels ) {
	  kiter = nkernels -1;
	  kstate = 1;
	}
	
      } while ( 0 == kstate );

      
      /* Okay, now we know what size kernel we need.  Now the 
	 fun beings.  We need to keep track of how much of each pixel is 
	 used in _all_ the kernels.  Then we need to go back
	 and smooth at the scale just determeined, but need to
	 rescale the data values */

      out_rad[outpix] = kiter; /* Save the kernel scale */
      
      out_sum[outpix] = sum;

      /* Now  we need to go thru the kernel again and 
	 sum up how much of the kernel is each point is using
       */
      pix = nkvals[kiter] -1;
      for (iy=ky[kiter];iy--;) {
	long ay;
	float *_kernel = kernel[kiter];
	ay = delta_yy + iy;
	if ( (ay < 0) ||( ay >= lAxes[1] )) {
	  pix-=kx[kiter]; /* skip one entire row */
	  continue;
	}
	
	for (ix=kx[kiter];ix--;) {
	  long ax;
	  ax = delta_xx + ix;
	  if (( ax < 0 ) || ( ax >= lAxes[0] )) {
	    pix--;
	    continue;
	  }
	  
	  /* Accumulate how many times each data pixel is used in
	     one of the kernels */
	  
	  if ( 0 == use_innrm  ) {
	    if (kernel_sum > 0 ) {
	      out_norm[ ax + ( ay * lAxes[0] )] += (_kernel[pix]/kernel_sum);
	    }
	  } else {
	    /* do nothing */
	  }
	  
	  pix--;
	  
	} /* end for ix */
      } /* end for iy */

    end_loop:
      
      outpix--;
      
    } /* end for xx */
  } /* end for yy */
  


  /* New data array which has been rescaled */
  outpix=( lAxes[0] * lAxes[1]) -1;
  norm_data = ( double*)calloc(( lAxes[0] * lAxes[1]), sizeof(double));
  for (yy=lAxes[1];yy--; ) {
    for (xx=lAxes[0];xx--;) {
      if ( out_norm[outpix] == 0 ) 
	out_norm[outpix] = 1;
      norm_data[outpix] = get_image_value( data, dt, xx, yy, lAxes, nullmask );
      norm_data[outpix] /= out_norm[outpix];
      outpix--;
    }
  }


  /* Okay ... now we have the scale and the normalization.
     Now we repeat w/ the above info.
  */
  outpix=( lAxes[0] * lAxes[1]) -1;

  if ( verbose > 1 ) {
    fprintf(stderr,"\nSecond iteration: computing final normalized values\n");
  }


  /* Okay, first loop over yy and xx which are the image axes */
  for (yy=lAxes[1];yy--; ) {
    long delta_yy;

    if ( verbose > 2 ) {
      float perc;
      perc = lAxes[1] - yy - 1;
      perc /= lAxes[1];
      perc *= 100;
      fprintf(stderr, "Percent complete: %5.2f%%\r", perc);
    }

    for (xx=lAxes[0];xx--;) {
      long pix;
      long delta_xx;

      double sum;
      double kernel_sum;

      long kiter = out_rad[outpix]; /* This is the saved value from above */
      float *_kernel;
      long _kx;
      long _ky;
      long _dkx;
      long _dky;
      long _nkvals;
      double nan_check;

      nan_check = get_image_value( data, dt, xx, yy, lAxes, nullmask );
      if ( ds_dNAN(nan_check) ) {
	out_rad[outpix] = nan_check;
	out_sum[outpix] = nan_check;
	out_norm[outpix] = nan_check;
	outdata[outpix] = nan_check;
	goto end_loop2;
      }



      /* Look familar?  cut-n-paste from above */
      
      _kernel = kernel[kiter];
      _kx = kx[kiter];
      _ky = ky[kiter];
      _dkx = dkx[kiter];
      _dky = dky[kiter];
      _nkvals = nkvals[kiter];
      
      delta_yy = yy - _dky;
      delta_xx = xx - _dkx;
      
      sum = 0;
      kernel_sum = 0;
      
      pix=_nkvals; /* Same game as w/ the output pixel */
      pix--;
      
      /* These are looping over the pixels in the kernel */
      for (iy=_ky;iy--;) {
	long ay;
	ay = delta_yy + iy;
	if ( (ay < 0) ||( ay >= lAxes[1] )) {
	  pix-=_kx;
	  continue;
	}
	
	for (ix=_kx;ix--;) {
	  long ax;
	  double dater;
	  ax = delta_xx + ix;
	  if (( ax < 0 ) || ( ax >= lAxes[0] )) {
	    pix--;
	    continue;
	  }
	  
	  dater = get_image_value( norm_data, dmDOUBLE, ax, ay, 
				   lAxes, nullmask );
	  if ( ds_dNAN(dater) ) {
	    pix--;
	    continue;
	  }
	  
	  sum += ( dater * _kernel[pix] );
	  kernel_sum += _kernel[pix];

	  pix--;
	  
	} /* end for ix */
      } /* end for iy */
      
      if ( 0 == kernel_sum ) {
	outdata[outpix] = sum;
      } else {
	outdata[outpix] = sum / kernel_sum;
      }
      
    end_loop2:

      outpix--;
      
    } /* end for xx */
  } /* end for yy */



  /* Now save all the data */

  put_param_hist_info( outBlock, "dmimgadapt", NULL, 0 );
  dmSetArray_d( outDesc, outdata, (lAxes[0]*lAxes[1]));
  dmImageClose(outBlock );

  if ( outRad ) {
    outpix=( lAxes[0] * lAxes[1]) -1;
    for (yy=lAxes[1];yy--; ) {
      for (xx=lAxes[0];xx--; ) {
	double dater = out_rad[outpix];
	long indx = dater;
	if ( ds_dNAN( dater ) ) {
	  out_rad[outpix] = dater;
	} else {
	  out_rad[outpix] = scales[ indx ];
	}
	outpix--;
      }
    }

    put_param_hist_info( outRad, "dmimgadapt", NULL, 0 );
    dmSetArray_d( outRadDesc, out_rad, (lAxes[0]*lAxes[1]));
    dmImageClose(outRad );
  }

  if ( outNorm ) { 
    put_param_hist_info( outNorm, "dmimgadapt", NULL, 0 );
    dmSetArray_d( outNormDesc, out_norm, (lAxes[0]*lAxes[1]));
    dmImageClose(outNorm );
  }

  if ( outSum ) { 
    put_param_hist_info( outSum, "dmimgadapt", NULL, 0 );
    dmSetArray_d( outSumDesc, out_sum, (lAxes[0]*lAxes[1]));
    dmImageClose(outSum );
  }


  return(0);

}








