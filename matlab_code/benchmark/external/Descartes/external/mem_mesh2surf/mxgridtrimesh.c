#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* MATLAB USAGE:
     *  [X,Y] = meshgrid(...,...);
     *  Z = mxgridtrimesh(F,V,X,Y);
     */
    
    int nx, ny, n, m, i, j, k;
    double *X, *Y, *Z, *F, *V;
    double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
    double w1, w2, w3, z;
    double minx, maxx, miny, maxy;
    int east, west, north, south;
    
    /* dimensions of input data */
    m = mxGetM(prhs[0]);
    n = mxGetM(prhs[1]);
    nx = mxGetN(prhs[2]);
    ny = mxGetM(prhs[2]);
    
    /* pointers to input data */
    F = mxGetPr(prhs[0]);
    V = mxGetPr(prhs[1]);
    X = mxGetPr(prhs[2]);
    Y = mxGetPr(prhs[3]);

    /* create mxArray and pointfor the output data */
    plhs[0] = mxCreateDoubleMatrix(ny, nx, mxREAL);

    /* pointer to the output data */
    Z = mxGetPr(plhs[0]);
    
    /* initialise output */
    for (i = 0; i < nx*ny; i++) Z[i] = mxGetNaN();
    
    /* consider every triangle, projected to the x-y plane and determine whether gridpoints lie inside */
    for (i = 0; i < m; i++) {
        v1x = V[(int)(F[i]-1)]; v1y = V[(int)(F[i]-1+n)]; v1z = V[(int)(F[i]-1+2*n)];
        v2x = V[(int)(F[m+i]-1)]; v2y = V[(int)(F[m+i]-1+n)]; v2z = V[(int)(F[m+i]-1+2*n)];
        v3x = V[(int)(F[2*m+i]-1)]; v3y = V[(int)(F[2*m+i]-1+n)]; v3z = V[(int)(F[2*m+i]-1+2*n)];
        /* we'll use the projected triangle's bounding box: of the form (minx,maxx) x (miny,maxy) */
        minx = v1x; if (v2x < minx) minx = v2x; if (v3x < minx) minx = v3x;
        maxx = v1x; if (v2x > maxx) maxx = v2x; if (v3x > maxx) maxx = v3x;
        /* find smallest x-grid value > minx, and largest x-grid value < maxx */
        east = mxGetNaN(); west = mxGetNaN();
        j = 0; while (j <= (nx-1)*ny && X[j] < minx) j = j + ny; if (j <= (nx-1)*ny) west = j/ny;
        j = (nx-1)*ny; while (j >= 0 && X[j] > maxx) j = j - ny; if (j >= 0) east = j/ny;
        /* if there are gridpoints strictly inside bounds (minx,maxx), continue */        
        if (!(mxIsNaN(west)) && !(mxIsNaN(east)) && east-west >= 0) {
            miny = v1y; if (v2y < miny) miny = v2y; if (v3y < miny) miny = v3y;
            maxy = v1y; if (v2y > maxy) maxy = v2y; if (v3y > maxy) maxy = v3y;
            /* find smallest y-grid value > miny, and largest y-grid value < maxy */
            north = mxGetNaN(); south = mxGetNaN();
            j = 0; while (j < ny && Y[j] < miny) j++; if (j < ny) north = j;
            j = ny-1; while (j >= 0 && Y[j] > maxy) j--; if (j >= 0) south = j;
            /* if, further, there are gridpoints strictly inside bounds (miny,maxy), continue */
            if (!(mxIsNaN(north)) && !(mxIsNaN(south)) && south-north >= 0)
                /* we now know that there might be gridpoints bounded by (west,east) x (north,south)
                   that lie inside the current triangle, so we'll test each of them */
                for (j = west; j <= east; j++)
                    for (k = north; k <= south; k++) {
                        /* calculate barycentric coordinates of gridpoint w.r.t. current (projected) triangle */
                        w1 = (v2y*v3x - v2x*v3y - v2y*X[j*ny+k] + v3y*X[j*ny+k] + v2x*Y[j*ny+k] - v3x*Y[j*ny+k])/(v1y*v2x - v1x*v2y - v1y*v3x + v2y*v3x + v1x*v3y - v2x*v3y);
                        w2 = (-(v3y*X[j*ny+k]) + v1y*(-v3x + X[j*ny+k]) + v1x*(v3y - Y[j*ny+k]) + v3x*Y[j*ny+k])/(v1y*(v2x - v3x) + v2y*v3x - v2x*v3y + v1x*(-v2y + v3y));
                        w3 = (v1y*v2x - v1x*v2y - v1y*X[j*ny+k] + v2y*X[j*ny+k] + v1x*Y[j*ny+k] - v2x*Y[j*ny+k])/(v1y*v2x - v1x*v2y - v1y*v3x + v2y*v3x + v1x*v3y - v2x*v3y);
                        if (w1 > 0 && w2 > 0 && w3 > 0) {
                            /* use barycentric coordinates to calculate z-value */
                            z = w1*v1z + w2*v2z + w3*v3z;
                            if (mxIsNaN(Z[j*ny+k]) || z > Z[j*ny+k]) Z[j*ny+k] = z;
                        }
                    }
        }
    }
}



