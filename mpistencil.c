/* Jesper Larsson Traff, January 2021, July-August 2021 */
/* Simple functionality test of singe interfaces and Cartesian naming 
   functionality */
/* Stencil examples from Parallel Computing paper */

#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

#include <mpi.h>

#include "mpiortholib.h"

// test some of the stuff
//#define DEBUG

#define BASICTEST
#define COLLTEST

// Example 9-point stencil with standard MPI as is
void stencil_orig(MPI_Comm comm)
{
  int d, p, n;
  MPI_Comm cart, cartcomm;
  int rank;
  int i, j;
  
  MPI_Comm_size(comm,&p);

  d = 2; // number of dimensions

  int order[d];
  int periodic[d];
  int reorder;

  for (i=0; i<d; i++) {
    order[i]    = 0;
    periodic[i] = 1;
  }
  MPI_Dims_create(p,d,order);
  reorder = 1; // reorder here?
  MPI_Cart_create(comm,d,order,periodic,reorder,&cart);
  MPI_Comm_rank(cart,&rank); // reorder could have taken place

  n = 25; // arbitrary choice of local matrix size

  double matrix[n+2][n+2];

  int t = 8;
  int target[] = { 0,1, 0,-1, -1,0,   1,0,
		  -1,1, 1,1,  1,-1, -1,-1};

  int sources[t], destinations[t];
  for (i=0; i<t; i++) {
    int vector[d];
    MPI_Cart_coords(cart,rank,d,vector);
    vector[0] += target[d*i];
    vector[1] += target[d*i+1];
    MPI_Cart_rank(cart,vector,&destinations[i]);
    MPI_Cart_coords(cart,rank,d,vector);
    vector[0] -= target[d*i];
    vector[1] -= target[d*i+1];
    MPI_Cart_rank(cart,vector,&sources[i]);
  }  

#ifdef DEBUG
  MPI_Comm_rank(comm,&rank);
  if (rank==2) {
    printf("Prop: ");
    for (i=0; i<t; i++) printf("%d ",destinations[i]);
    printf("\n");
  }
#endif
  
  reorder = 1; // or reorder here?     
  MPI_Dist_graph_create_adjacent(cart,
				 t,destinations,MPI_UNWEIGHTED,
				 t,destinations,MPI_UNWEIGHTED,
				 MPI_INFO_NULL,reorder,&cartcomm);

  MPI_Datatype ROW, COL, COR;
  
  MPI_Type_contiguous(n,MPI_DOUBLE,&ROW);
  MPI_Type_commit(&ROW);
  MPI_Type_vector(n,1,n+2,MPI_DOUBLE,&COL);
  MPI_Type_commit(&COL);
  MPI_Type_contiguous(1,MPI_DOUBLE,&COR);
  MPI_Type_commit(&COR);

  int sendcount[t], recvcount[t];
  MPI_Aint senddisp[t], recvdisp[t];
  MPI_Datatype sendtype[t], recvtype[t];

  // by using datatypes, all counts 1
  for (i=0; i<t; i++) {
    sendcount[i] = 1;
    recvcount[i] = 1;
  }

  senddisp[0] = 1*(n+2)+n;     sendtype[0] = COL;
  recvdisp[0] = 1*(n+2)+n+1;   recvtype[0] = COL;
  senddisp[1] = 1*(n+2)+1;     sendtype[1] = COL;
  recvdisp[1] = 1*(n+2);       recvtype[1] = COL;
  senddisp[2] = 1*(n+2)+1;     sendtype[2] = ROW;
  recvdisp[2] = 1;             recvtype[2] = ROW;
  senddisp[3] = n*(n+2)+1;     sendtype[3] = ROW;
  recvdisp[3] = (n+1)*(n+2)+1; recvtype[3] = ROW;

  senddisp[4] = n*(n+2)+1;       sendtype[4] = COR;
  recvdisp[4] = (n+1)*(n+2);     recvtype[4] = COR;
  senddisp[5] = 1*(n+2)+n;       sendtype[5] = COR;
  recvdisp[5] = n+1;             recvtype[5] = COR;
  senddisp[6] = n*(n+2)+n;       sendtype[6] = COR;
  recvdisp[6] = (n+1)*(n+2)+n+1; recvtype[6] = COR;
  senddisp[7] = 1*(n+2)+1;       sendtype[7] = COR;
  recvdisp[7] = 0;               recvtype[7] = COR;
    
  // byte offsets
  for (i=0; i<t; i++) {
    senddisp[i] *= sizeof(double);
    recvdisp[i] *= sizeof(double);
  }
  
  short iterate = 1;
  int iter = 0;
  while (iterate) {
    // compute ...
    for (i=1; i<n+1; i++) {
      for (j=1; j<n+1; j++) {
	matrix[i][j] = 2*matrix[i][j]; // something more sensible
      }
    }

    // update
    MPI_Neighbor_alltoallw(matrix,sendcount,senddisp,sendtype,
			   matrix,recvcount,recvdisp,recvtype,cartcomm);
    if (rank==0) printf("Iter orig %d\n",iter);
      
    // converged?
    int local = 1; // local check  
    MPI_Allreduce(&local,&iterate,1,MPI_SHORT,MPI_LAND,cartcomm);
    iter++;
    if (iter>10) iterate = 0;
  }

  MPI_Comm_free(&cart);
  MPI_Comm_free(&cartcomm);

  MPI_Type_free(&ROW);
  MPI_Type_free(&COL);
  MPI_Type_free(&COR);
}

// Example 9-point stencil with with proposed new interfaces
void stencil_prop(MPI_Comm comm)
{
  int d, p, pp, n;
  MPI_Comm cartcomm, basecomm;
  int r;
  int i, j;
  
  MPI_Comm_rank(comm,&r);
  MPI_Comm_size(comm,&p);

  d = 2; // number of dimensions

  int order[d];
  int periodic[d];
  int reorder;

  for (i=0; i<d; i++) {
    order[i]    = 0;
    periodic[i] = 1;
  }
  MPI_Dims_create(p,d,order);
  TUW_Cart_name(comm,d,TUW_ROWMAJOR,order,periodic,&pp);

  n = 25; // arbitrary choice of local matrix size
  
  double matrix[n+2][n+2];

  int t = 8; 
  int target[] = { 0,1, 0,-1, -1,0,   1,0,
                  -1,1, 1,1,   1,-1, -1,-1};  

  int destinations[t];
  TUW_Cart_allranks_relative(comm,r,t,target,destinations);
  
  reorder = 1; // reorder?                 
  MPI_Dist_graph_create_adjacent(comm,
				 t,destinations,MPI_UNWEIGHTED,
				 t,destinations,MPI_UNWEIGHTED,
				 MPI_INFO_NULL,reorder,&cartcomm);
  // sanity check
  assert(tuw_isCartesian(comm,cartcomm));

  TUW_Comm_base(cartcomm,&basecomm);

  MPI_Datatype ROW, COL, COR;
  
  MPI_Type_contiguous(n,MPI_DOUBLE,&ROW);
  MPI_Type_commit(&ROW);
  MPI_Type_vector(n,1,n+2,MPI_DOUBLE,&COL);
  MPI_Type_commit(&COL);
  MPI_Type_contiguous(1,MPI_DOUBLE,&COR);
  MPI_Type_commit(&COR);

  int sendcount[t], recvcount[t];
  MPI_Aint senddisp[t], recvdisp[t];
  MPI_Datatype sendtype[t], recvtype[t];

  // by using datatypes, all counts 1
  for (i=0; i<t; i++) {
    sendcount[i] = 1;
    recvcount[i] = 1;
  }

  senddisp[0] = 1*(n+2)+n;     sendtype[0] = COL;
  recvdisp[0] = 1*(n+2)+n+1;   recvtype[0] = COL;
  senddisp[1] = 1*(n+2)+1;     sendtype[1] = COL;
  recvdisp[1] = 1*(n+2);       recvtype[1] = COL;
  senddisp[2] = 1*(n+2)+1;     sendtype[2] = ROW;
  recvdisp[2] = 1;             recvtype[2] = ROW;
  senddisp[3] = n*(n+2)+1;     sendtype[3] = ROW;
  recvdisp[3] = (n+1)*(n+2)+1; recvtype[3] = ROW;

  senddisp[4] = n*(n+2)+1;       sendtype[4] = COR;
  recvdisp[4] = (n+1)*(n+2);     recvtype[4] = COR;
  senddisp[5] = 1*(n+2)+n;       sendtype[5] = COR;
  recvdisp[5] = n+1;             recvtype[5] = COR;
  senddisp[6] = n*(n+2)+n;       sendtype[6] = COR;
  recvdisp[6] = (n+1)*(n+2)+n+1; recvtype[6] = COR;
  senddisp[7] = 1*(n+2)+1;       sendtype[7] = COR;
  recvdisp[7] = 0;               recvtype[7] = COR;

  // byte offsets
  for (i=0; i<t; i++) {
    senddisp[i] *= sizeof(double);
    recvdisp[i] *= sizeof(double);
  }
  
  short iterate = 1;
  int iter = 0;
  while (iterate) {
    // compute ...
    for (i=1; i<n+1; i++) {
      for (j=1; j<n+1; j++) {
	matrix[i][j] = 2*matrix[i][j]; // something more sensible
      }
    }
    
    // update
    TUW_Alltoallw(matrix,sendcount,senddisp,sendtype,
		  matrix,recvcount,recvdisp,recvtype,cartcomm);
    if (r==0) printf("Iter prop %d\n",iter);
    
    // converged?
    int local = 1; // local check  
    MPI_Allreduce(&local,&iterate,1,MPI_SHORT,MPI_LAND,basecomm);
    iter++;
    if (iter>10) iterate = 0;
  }

  MPI_Comm_free(&cartcomm);
  MPI_Comm_free(&basecomm);
  
  MPI_Type_free(&ROW);
  MPI_Type_free(&COL);
  MPI_Type_free(&COR);
}

// Toy game-of-life with new, sparse allreduce
void life_prop(MPI_Comm comm)
{
  int d, p, pp, n;
  MPI_Comm cartcomm, basecomm;
  int r;
  int i;

  d = 2; // number of dimensions

  int order[d];
  int periodic[d];
  int reorder;

  MPI_Comm_rank(comm,&r);
  MPI_Comm_size(comm,&p);
  
  for (i=0; i<d; i++) {
    order[i]    = 0;
    periodic[i] = 1;
  }
  MPI_Dims_create(p,d,order);
  TUW_Cart_name(comm,d,TUW_ROWMAJOR,order,periodic,&pp);

  int t;
  TUW_Cart_neighbors_count(comm,r,TUW_CHEBYCHEV_DISTANCE,1,1,&t);
  printf("t=%d\n",t);
  assert(t==8);
  
  int ranks[t];
  TUW_Cart_neighbors(comm,r,TUW_CHEBYCHEV_DISTANCE,1,1,t,ranks);

  // previous hack
  /*
  int target2[] = { 0,1, 0,-1, -1,0,   1,0,
		    -1,1, 1,1,   1,-1, -1,-1 };
  TUW_Cart_allranks_relative(comm,r,t,target2,ranks);
  // filter out MPI_PROC_NULL
  */

  reorder = 1; // reorder?                 
  MPI_Dist_graph_create_adjacent(comm,
				 t,ranks,MPI_UNWEIGHTED,
				 t,ranks,MPI_UNWEIGHTED,
				 MPI_INFO_NULL,reorder,&cartcomm);
  TUW_Comm_base(cartcomm,&basecomm);

  short state, live;

  state = (r%3==0) ? 1 : 0; // find more interesting initial configuration
  
  int iter = 0;
  short forever = 1;
  while (forever) {
    TUW_Allreduce(&state,&live,1,MPI_SHORT,MPI_SUM,cartcomm);
    if (live<2||live>3) state = 0; else if (live==3) state = 1;

    MPI_Allreduce(&state,&forever,1,MPI_SHORT,MPI_LOR,basecomm);
    if (r==0||r==p-1)
      printf("Rank %d, iter: %d state %hd live %hd forever %hd\n",
	     r,iter,state,live,forever);
    iter++;
    if (iter==10) forever = 0;
  }

  MPI_Comm_free(&cartcomm);
  MPI_Comm_free(&basecomm);
}

int main(int argc, char *argv[])
{
  int rank, size;
  int d;
  int i, j, k;
  int p;
  int reorder;
  
  MPI_Init(&argc,&argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  // generate a simple cart-named dist graph, get the base comm
  // use the collectives

  // verify against MPI_Cart functionality

#ifdef BASICTEST
  for (d=1; d<size; d++) {
    if (d>5) break;
    
    int order[d];
    int periodic[d];
    int coordinates[d];
    int flag;
    int dd, ss;
    
    for (i=0; i<d; i++) {
      order[i]    = 0;
      periodic[i] = 1;
    }
    MPI_Dims_create(size,d,order);

    if (rank==0) printf("d: %d\n",d);

    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD,d,order,periodic,0,&cart);
    int cartcoords[d];
    
    TUW_Cart_name(MPI_COMM_WORLD,d,TUW_ROWMAJOR,order,periodic,&p);
    assert(p==size);
    // check for required, consistent naming
    TUW_Cart_testall(MPI_COMM_WORLD,&flag,&dd,&ss);
    printf("Rank %d d %d %d s %d %d\n",rank,d,dd,p,ss);
    assert(flag);
    assert(dd==d);
    assert(ss==p);

    for (i=0; i<size; i++) {
      TUW_Cart_coordinates(MPI_COMM_WORLD,i,coordinates);
      MPI_Cart_coords(cart,i,d,cartcoords);

      for (j=0; j<d; j++) assert(coordinates[j]==cartcoords[j]);
#ifdef DEBUG
      if (rank==0) {
	printf("rank %d: ",i);
	for (j=0; j<d; j++) {
	  printf("%d==%d (%d) ",coordinates[j],cartcoords[j],order[j]);
	}
	printf("\n");
      }
#endif

      TUW_Cart_rank(MPI_COMM_WORLD,coordinates,&j);
      assert(j==i);
    }

    int shift[d];
    for (j=0; j<d; j++) shift[j] = 0;
    for (j=0; j<d; j++) {
      shift[j] = j;
      int inrank, outrank;
      TUW_Cart_relative_shift(MPI_COMM_WORLD,rank,shift,&inrank,&outrank);
      shift[j] = 0; // next dimension
      
      int source, dest;
      MPI_Cart_shift(cart,j,j,&source,&dest);

#ifdef DEBUG
      //if (inrank!=source||outrank!=dest)
      printf("Rank %d, shift %d: in %d==%d out %d==%d\n",
	rank,j,inrank,source,outrank,dest);
#endif
      assert(inrank==source);
      assert(outrank==dest);
    }

    for (j=1; j<=d; j++) {
      int n;
      TUW_Cart_neighbors_count(MPI_COMM_WORLD,rank,
			       TUW_MANHATTAN_DISTANCE,1,j,&n);
      if (rank==size-1) {
	printf("Rank %d: Manhattan neighbors %d with depth %d\n",rank,n,j);
      }
    }
    for (j=1; j<=d; j++) {
      int n;
      TUW_Cart_neighbors_count(MPI_COMM_WORLD,rank,
			       TUW_CHEBYCHEV_DISTANCE,1,j,&n);
      if (rank==size-1) {
	printf("Rank %d: Chebychev neighbors %d with depth %d\n",rank,n,j);
      }
    }

    if (d==2) {
      assert(d>=2);
      int n;
      TUW_Cart_stencil_count(MPI_COMM_WORLD,TUW_CHEBYCHEV_DISTANCE,1,&n);
      int target[d*n];
      TUW_Cart_stencil(MPI_COMM_WORLD,
		       TUW_CHEBYCHEV_DISTANCE,1,n,target);
      if (rank==size/2) {
	printf("Chebychev:\n");
	for (i=0; i<n; i++) {
	  printf("(%d,%d) ",target[d*i],target[d*i+1]);
	}
	printf("\n");
      }
    }
    if (d==3) {
      assert(d>2);
      int n;
      TUW_Cart_stencil_count(MPI_COMM_WORLD,TUW_MANHATTAN_DISTANCE,2,&n);
      int target[d*n];
      TUW_Cart_stencil(MPI_COMM_WORLD,
		       TUW_MANHATTAN_DISTANCE,2,n,target);
      if (rank==size-1) {
	printf("Manhattan:\n");
	for (i=0; i<n; i++) {
	  printf("(%d,%d,%d) ",target[d*i],target[d*i+1],target[d*i+2]);
	}
	printf("\n");
      }
    }

    MPI_Comm cartsub;
    int sizesub, expsub;
    int keepdimension[d];
    for (i=0; i<d; i++) keepdimension[i] = 1;
    TUW_Cart_create_sub(MPI_COMM_WORLD,keepdimension,&cartsub);
    if (cartsub!=MPI_COMM_NULL) {
      MPI_Comm_size(cartsub,&sizesub);
      expsub = 1;
      for (i=0; i<d; i++) if (keepdimension[i]) expsub *= order[i];
      assert(sizesub==expsub);
      MPI_Comm_free(&cartsub);
    }
    keepdimension[0] = 0;
    TUW_Cart_create_sub(MPI_COMM_WORLD,keepdimension,&cartsub);
    if (cartsub!=MPI_COMM_NULL) {
      MPI_Comm_size(cartsub,&sizesub);
      expsub = 1;
      for (i=0; i<d; i++) if (keepdimension[i]) expsub *= order[i];
      assert(sizesub==expsub);
      MPI_Comm_free(&cartsub);
    }
    keepdimension[d-1] = 0;
    TUW_Cart_create_sub(MPI_COMM_WORLD,keepdimension,&cartsub);
    if (cartsub!=MPI_COMM_NULL) {
      MPI_Comm_size(cartsub,&sizesub);
      expsub = 1;
      for (i=0; i<d; i++) if (keepdimension[i]) expsub *= order[i];
      assert(sizesub==expsub);
      MPI_Comm_free(&cartsub);
    }
    
    MPI_Comm_free(&cart);
  }
#endif

#ifdef COLLTEST
  reorder = 0;
  int source[size];
  int target[size];
  
  MPI_Comm distcomm;

  // set up a dist graph
  for (j=0; j<rank; j++) target[j] = j;
  for (j=0; j<size-1-rank; j++) source[j] = j+rank+1;
  
  MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
				 size-1-rank,source,MPI_UNWEIGHTED,
				 rank,target,MPI_UNWEIGHTED,
				 MPI_INFO_NULL,reorder,&distcomm);
  // sanity check
  assert(!tuw_isCartesian(MPI_COMM_WORLD,distcomm));
  
  // try the new collectives

  TUW_Barrier(distcomm);
  TUW_Barrier(distcomm);
  
  for (i=0; i<size; i++) {
    double buf[100];
    if (rank==i) {
      for (j=0; j<100; j++) buf[j] = (double)rank;
    } else {
      for (j=0; j<100; j++) buf[j] = (double)-1;
    }
    
    TUW_Bcast(buf,100,MPI_DOUBLE,i,distcomm);
    for (k=0; k<size-1-rank; k++) {
      if (source[k]==i) {
	for (j=0; j<size; j++) assert(buf[j]==(double)i);
	break;
      }
    }
    if (rank!=i&&k==size-1-rank) {
      for (j=0; j<size; j++) assert(buf[j]==(double)-1);
    }
  }

  for (i=0; i<size; i++) {
    double rbuf[size*10];
    double sbuf[10];
    if (rank==i) {
      for (j=0; j<size*10; j++) rbuf[j] = (double)-1;
      for (j=0; j<10; j++) sbuf[j] = (double)rank+j;
    } else {
      for (j=0; j<10; j++) sbuf[j] = (double)rank+j;
    }
    
    TUW_Gather(sbuf,10,MPI_DOUBLE,rbuf,10,MPI_DOUBLE,i,distcomm);
    if (rank==i) {
      for (k=0; k<size-1-rank; k++) {
	for (j=0; j<10; j++) assert(rbuf[k*10+j]==(double)source[k]+j);
      }
    }
  }
  
  for (i=0; i<size; i++) {
    int cin, cout;
    cin = -1;
    cout = 0;
    TUW_Reduce(&cin,&cout,1,MPI_INT,MPI_SUM,i,distcomm);
    if (rank==i) assert(cout==-(size-1-rank));
  }
  
  int cin, cout;
  cin = 55;
  cout = 0;
  
  TUW_Allreduce(&cin,&cout,1,MPI_INT,MPI_SUM,distcomm);
  assert(cout==55*(size-1-rank));

  // even-odd split communicator, map subgraph
  MPI_Comm subcomm, subdistcomm;
  MPI_Comm_split(MPI_COMM_WORLD,rank%2,0,&subcomm);

  int subrank;
  MPI_Comm_rank(subcomm,&subrank);
  
  int s, ss, t, tt;
  int w, ww;
  TUW_Comm_subgraph(distcomm,subcomm,&subdistcomm);
  MPI_Dist_graph_neighbors_count(distcomm,&s,&t,&w);
  MPI_Dist_graph_neighbors_count(subdistcomm,&ss,&tt,&ww);
  printf("Rank %d: weigth %d %d sources %d %d targets %d %d\n",
	 rank,w,ww, s,ss,t,tt);

  // try some collectives on the distributed graph subcommunicators

  TUW_Barrier(subdistcomm);
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank==0) {
    MPI_Dist_graph_neighbors(distcomm,
			     s,source,MPI_UNWEIGHTED,t,target,MPI_UNWEIGHTED);
    printf("Rank %d sources: ",rank);
    for (i=0; i<s; i++) {
      printf("%d ",source[i]);
    }
    printf("\n");
    printf("Rank %d targets: ",rank);
    for (i=0; i<t; i++) {
      printf("%d ",target[i]);
    }
    printf("\n");

    MPI_Dist_graph_neighbors(subdistcomm,
			     ss,source,MPI_UNWEIGHTED,tt,target,MPI_UNWEIGHTED);
    printf("SubRank %d sources: ",subrank);
    for (i=0; i<ss; i++) {
      printf("%d ",source[i]);
    }
    printf("\n");
    printf("SubRank %d targets: ",subrank);
    for (i=0; i<tt; i++) {
      printf("%d ",target[i]);
    }
    printf("\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank==size-1) {
    MPI_Dist_graph_neighbors(distcomm,
			     s,source,MPI_UNWEIGHTED,t,target,MPI_UNWEIGHTED);
    printf("Rank %d sources: ",rank);
    for (i=0; i<s; i++) {
      printf("%d ",source[i]);
    }
    printf("\n");
    printf("Rank %d targets: ",rank);
    for (i=0; i<t; i++) {
      printf("%d ",target[i]);
    }
    printf("\n");

    MPI_Dist_graph_neighbors(subdistcomm,
			     ss,source,MPI_UNWEIGHTED,tt,target,MPI_UNWEIGHTED);
    printf("SubRank %d sources: ",subrank);
    for (i=0; i<ss; i++) {
      printf("%d ",source[i]);
    }
    printf("\n");
    printf("SubRank %d targets: ",subrank);
    for (i=0; i<tt; i++) {
      printf("%d ",target[i]);
    }
    printf("\n");
  }
  
  MPI_Comm_free(&distcomm);
  MPI_Comm_free(&subcomm);
  MPI_Comm_free(&subdistcomm);
#endif
  
  // larger examples
  
  MPI_Comm stencomm, lifecomm;
  MPI_Comm_dup(MPI_COMM_WORLD,&stencomm);
  MPI_Comm_dup(MPI_COMM_WORLD,&lifecomm);

  stencil_orig(stencomm);
  stencil_prop(stencomm);
  
  life_prop(lifecomm);
  
  MPI_Comm_free(&stencomm);
  MPI_Comm_free(&lifecomm);
  
  MPI_Finalize();
  
  return 0;
}
