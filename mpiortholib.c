/* Jesper Larsson Traff, November 2020, January 2021, June 2021, August 2021 */
/* Prototype library and support for single collective interfaces for
   fully connected, bi-partite, and directed graph topologies in support
   of Parallel Computing paper */

/* Experimental code for illustrative purposes, no warranty... */
/* Rudimentary error checking only */

#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

#include <mpi.h>

#include "mpiortholib.h"

typedef struct {
  int d;
  int dimorder;
  int *order;
  int *periodic;
  int size;
} nameattr;

static int namedel(MPI_Comm comm, int keyval, void *attr, void *s) {
  nameattr *cartname = (nameattr*)attr;

  free(cartname->order);
  free(cartname->periodic);
  free(cartname);
  
  return MPI_SUCCESS;
}

// We let an attribute keyval be created by MPI, instead of statically chosen

static int namekey() {
  // hidden key value for type attributes
  static int namekeyval = MPI_KEYVAL_INVALID;

  if (namekeyval == MPI_KEYVAL_INVALID) {
    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,namedel,&namekeyval,NULL);
  }

  return namekeyval;
}

// Get the (fully connected) base communicator:
// get rid of topology and attributes

// Note: library implementation has collective, non-local semantics

int TUW_Comm_base(MPI_Comm comm, MPI_Comm *basecomm)
{
  MPI_Group group;
  
  MPI_Comm_group(comm,&group); // local group for an inter-communicator
  MPI_Comm_create(comm,group,basecomm); // Warning: collective, non-local
  MPI_Group_free(&group);
  
  return MPI_SUCCESS;
}

// Attach Cartesian naming to comm
// ROW/COLMAJOR follow the (weird) MPI convention

int TUW_Cart_name(MPI_Comm comm,
		  int d, int dimorder, int order[], int periodic[], int *size)
{
  nameattr *cartname;
  int flag;
  int i;
  int commsize;
  
  // check for valid Cartesian naming
  MPI_Comm_size(comm,&commsize);
  *size = 1;
  for (i=0; i<d; i++) *size *= order[i];
  if (commsize<*size) return MPI_ERR_COMM;
  if (dimorder!=TUW_ROWMAJOR&&dimorder!=TUW_COLMAJOR) return MPI_ERR_DIMS;
  
  MPI_Comm_get_attr(comm,namekey(),&cartname,&flag);
  if (flag) MPI_Comm_delete_attr(comm,namekey());

  cartname = (nameattr*)malloc(sizeof(nameattr));
  assert(cartname!=NULL);
  cartname->d = d;
  cartname->dimorder = dimorder;
  cartname->order = malloc(d*sizeof(int));
  assert(cartname->order!=NULL);
  cartname->periodic = malloc(d*sizeof(int));
  assert(cartname->periodic!=NULL);
  cartname->size = *size;

  for (i=0; i<d; i++) {
    cartname->order[i] =  order[i];
    cartname->periodic[i] = periodic[i];
  }
  
  MPI_Comm_set_attr(comm,namekey(),cartname);

  return MPI_SUCCESS;
}

// Query function: Is Cartesian naming attached?

int TUW_Cart_test(MPI_Comm cart, int *flag, int *d, int *size)
{
  nameattr *cartname;

  MPI_Comm_get_attr(cart,namekey(),&cartname,flag);
  if (*flag) {
    *d =    cartname->d;
    *size = cartname->size;
  } else {
    *d =    -1;
    *size = -1;
  }

  return MPI_SUCCESS;
}

// Collective test function, are all processes using same naming?
// (helper functionality currently not intended as part of single interface
// proposal)

int TUW_Cart_testall(MPI_Comm cart, int *flag, int *d, int *size)
{
  nameattr *cartname;
  int rank;
  int i;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,flag);
  
  MPI_Allreduce(MPI_IN_PLACE,flag,1,MPI_INT,MPI_LAND,cart);
  if (!*flag) return MPI_ERR_TOPOLOGY;

  *d = cartname->d;
  MPI_Bcast(d,1,MPI_INT,0,cart);
  
  *flag = (*d==cartname->d);
  MPI_Allreduce(MPI_IN_PLACE,flag,1,MPI_INT,MPI_LAND,cart);
  if (!*flag) return MPI_ERR_TOPOLOGY;

  MPI_Comm_rank(cart,&rank);
  if (rank==0) {
    MPI_Bcast(cartname->order,   *d,MPI_INT,0,cart);
    MPI_Bcast(cartname->periodic,*d,MPI_INT,0,cart);
  } else {
    int order[*d];
    int periodic[*d];
    
    MPI_Bcast(order,*d,MPI_INT,0,cart);
    for (i=0; i<*d; i++) if (order[i]!=cartname->order[i]) break;
    if (i<*d) *flag = 0;
    MPI_Bcast(periodic,*d,MPI_INT,0,cart);
    for (i=0; i<*d; i++) if (periodic[i]!=cartname->periodic[i]) break;
    if (i<*d) *flag = 0;
  }

  MPI_Allreduce(MPI_IN_PLACE,flag,1,MPI_INT,MPI_LAND,cart);
  if (!*flag) return MPI_ERR_TOPOLOGY;

  *size = cartname->size;

  return MPI_SUCCESS;
}

// Get parameters of Cartesian naming back

int TUW_Cart_get(MPI_Comm cart, int *dimorder,
		 int maxd, int order[], int periodic[])
{
  nameattr *cartname;
  int flag;
  int i;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  *dimorder = cartname->dimorder;
  for (i=0; i<maxd; i++) {
    order[i]  = cartname->order[i];
    periodic[i] = cartname->periodic[i];
  }

  return MPI_SUCCESS;
}

// Absolute navigation: Check and compute

// From coordinate vector to rank

int TUW_Cart_rank(MPI_Comm cart, int coordinates[], int *rank)
{
  nameattr *cartname;
  int flag;
  int i;
  int a, c;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;
  
  if (cartname->dimorder==TUW_ROWMAJOR) {
    *rank = 0;
    a = 1;
    for (i=cartname->d-1; i>=0; i--) {
      if (!cartname->periodic[i]) {
	if (coordinates[i]<0||coordinates[i]>=cartname->order[i])
	  return MPI_ERR_DIMS;
      }
      // compute row major rank
      c = coordinates[i];
      while (c<0) c += cartname->order[i];
      c = c%cartname->order[i];
      *rank += a*c; 
      a *= cartname->order[i];
    }
  } else if (cartname->dimorder==TUW_COLMAJOR) {
    *rank = 0;
    a = 1;
    for (i=0; i<cartname->d; i++) {
      if (!cartname->periodic[i]) {
	if (coordinates[i]<0||coordinates[i]>=cartname->order[i])
	  return MPI_ERR_DIMS;
      }
      // compute col major rank
      c = coordinates[i];
      while (c<0) c += cartname->order[i];
      c = c%cartname->order[i];
      *rank += a*c;
      a *= cartname->order[i];
    }
  } else {
    return MPI_ERR_TOPOLOGY; 
  }
  
  return MPI_SUCCESS;
}

int TUW_Cart_coordinates(MPI_Comm cart, int rank, int coordinates[])
{
  nameattr *cartname;
  int flag;
  int i;
  int a;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  if (rank<0||rank>=cartname->size) return MPI_ERR_RANK;
  // should MPI_PROC_NULL be allowed here?
  
  if (cartname->dimorder==TUW_ROWMAJOR) {
    a = 1;
    for (i=0; i<cartname->d; i++) a *= cartname->order[i];
    for (i=0; i<cartname->d; i++) {
      a /= cartname->order[i];
      coordinates[i] = rank/a;
      rank = rank%a;
    }
  } else if (cartname->dimorder==TUW_COLMAJOR) {
    a = 1;
    for (i=0; i<cartname->d; i++) a *= cartname->order[i];
    for (i=cartname->d-1; i>=0; i--) {
      a /= cartname->order[i];
      coordinates[i] = rank/a;
      rank = rank%a;
    }
  } else {
    return MPI_ERR_TOPOLOGY; 
  }
  
  return MPI_SUCCESS;
}

// Relative (vector offset) navigation
// For non-existing neighbor in non-periodic grid MPI_PROC_NULL

int TUW_Cart_relative_rank(MPI_Comm cart, int source, int relative[],
			   int *rank)
{
  nameattr *cartname;
  int flag;
  int i;
  int d;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  if (source<0||source>=cartname->size) return MPI_ERR_RANK;
  // should MPI_PROC_NULL be allowed here?

  d = cartname->d;

  int coordinates[d];
  TUW_Cart_coordinates(cart,source,coordinates);

  for (i=0; i<d; i++) coordinates[i] = coordinates[i]+relative[i];
  
  TUW_Cart_rank(cart,coordinates,rank);

  return MPI_SUCCESS;
}

int TUW_Cart_relative_coordinates(MPI_Comm cart, int source, int dest,
				  int relative[])
{
  nameattr *cartname;
  int flag;
  int d;
  int i;
  int r;
  int mpi_code;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;

  if (source<0||source>=cartname->size) return MPI_ERR_RANK;
  // should MPI_PROC_NULL be allowed here?
  if (dest<0||dest>=cartname->size) return MPI_ERR_RANK;
  // should MPI_PROC_NULL be allowed here?

  int coordinates[d];
  TUW_Cart_coordinates(cart,source,coordinates);
  TUW_Cart_coordinates(cart,dest,relative);

  for (i=0; i<d; i++) relative[i] = relative[i]-coordinates[i];

  return MPI_SUCCESS;
}

int TUW_Cart_relative_shift(MPI_Comm cart, int rank, int relative[],
			    int *inrank, int *outrank)
{
  nameattr *cartname;
  int flag;
  int d;
  int i;
  int mpi_code;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  if (rank<0||rank>=cartname->size) return MPI_ERR_RANK;
  // should MPI_PROC_NULL be allowed here?

  mpi_code = TUW_Cart_relative_rank(cart,rank,relative,outrank);
  if (mpi_code!=MPI_SUCCESS) return mpi_code;
  
  d = cartname->d;
  
  int negative[d];
  
  for (i=0; i<d; i++) negative[i] = -relative[i];
  return TUW_Cart_relative_rank(cart,rank,negative,inrank);
}

// Should these functions remove MPI_PROC_NULL?
int TUW_Cart_allranks(MPI_Comm cart, int n, int coordinates[], int ranks[])
{
  nameattr *cartname;
  int flag;
  int d;
  int i;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;
  for (i=0; i<n; i++) {
    TUW_Cart_rank(cart,coordinates+i*d,&ranks[i]);
  }
  
  return MPI_SUCCESS;
}

int TUW_Cart_allranks_relative(MPI_Comm cart, int source,
			       int n, int relatives[],
			       int ranks[])
{
  nameattr *cartname;
  int flag;
  int d;
  int i;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;
  for (i=0; i<n; i++) {
    TUW_Cart_relative_rank(cart,source,relatives+i*d,&ranks[i]);
  }
  
  return MPI_SUCCESS;
}

// These should definitely remove MPI_PROC_NULL

int TUW_Cart_neighbors_count(MPI_Comm cart, int rank,
			     int metric, int shadow, int depth, int *size)
{
  nameattr *cartname;
  int flag;
  int d;
  int s, ss;

  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;

  TUW_Cart_stencil_count(cart,metric,depth,&s);
  int coorddepth[s*d];
  TUW_Cart_stencil(cart,metric,depth,s,coorddepth);

  if (shadow>0) {
    TUW_Cart_stencil_count(cart,metric,shadow-1,&ss);
    int coordshadow[ss*d];
    TUW_Cart_stencil(cart,metric,shadow-1,ss,coordshadow);

    int coord[s*d];
    TUW_Cart_stencil_diff(cart,s,coorddepth,ss,coordshadow,size,coord);
  } else *size = s;
  // for no-periodic grids, remove some
  
  return MPI_SUCCESS;  
}

// maxsize ignored
int TUW_Cart_neighbors(MPI_Comm cart, int rank,
		       int metric, int shadow, int depth,
		       int maxsize, int neighbors[])
{
  nameattr *cartname;
  int flag;
  int d;
  int s, ss;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;

  TUW_Cart_stencil_count(cart,metric,depth,&s);

  int coord[s*d];
  if (shadow==0) {
    TUW_Cart_stencil(cart,metric,depth,s,coord);
  } else {
    int coorddepth[s*d];
    TUW_Cart_stencil(cart,metric,depth,s,coorddepth);

    TUW_Cart_stencil_count(cart,metric,shadow-1,&ss);
    int coordshadow[ss*d];
    TUW_Cart_stencil(cart,metric,shadow-1,ss,coordshadow);

    TUW_Cart_stencil_diff(cart,s,coorddepth,ss,coordshadow,&s,coord);
  }
  MPI_Comm_rank(cart,&rank);
  TUW_Cart_allranks_relative(cart,rank,s,coord,neighbors);
  // this will/should filter out MPI_PROC_NULL
  
  return MPI_SUCCESS;
}

// Stencil functionality creates stencils as list of relative coordinates
// Simple functionality to put stencils together and form differences

// Cartesian neighborhood functionality creates lists of neighbors (ranks)
// MPI_PROC_NULL neighbors removed; to be used directly as input to
// MPI_Dist_graph_create functionality

// Note: only thing used from the naming scheme is d
// Functionality might as well be totally outside of MPI/TUW

int TUW_Cart_stencil_count(MPI_Comm cart, int metric, int depth, int *size)
{
  nameattr *cartname;
  int flag;
  int i;
  int d;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;

  if (metric==TUW_MANHATTAN_DISTANCE) {
    // von Neumann neighborhood
    *size = (2*depth+1)*d-(d-1);
  } else if (metric==TUW_CHEBYCHEV_DISTANCE) {
    // Moore neighborhood
    * size = 1;
    for (i=0; i<d; i++) *size *= (2*depth+1);
  } else {
    return MPI_ERR_UNKNOWN;
  }
  
  return MPI_SUCCESS;  
}

int TUW_Cart_stencil(MPI_Comm cart, int metric, int depth,
		     int maxsize, int coordinates[])
{
  nameattr *cartname;
  int flag;
  int d;

  int i, j, k, c;

  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;
  
  if (metric==TUW_MANHATTAN_DISTANCE) {
    // von Neumann neighborhood

    for (i=0; i<d; i++) coordinates[i] = 0; // center
    k = 1;
    for (i=0; i<d; i++) {
      for (c=-depth; c<=depth; c++) {
	if (c==0) continue;
	for (j=0; j<i; j++) coordinates[k*d+j] = 0;
	coordinates[k*d+i] = c;
	for (j=i+1; j<d; j++) coordinates[k*d+j] = 0;
	k++;
      }
    }
  } else if (metric==TUW_CHEBYCHEV_DISTANCE) {
    // Moore neighborhood
    int s, ss, ns;
    int dd;

    dd = 2*depth+1;
    
    s = 1;
    for (i=0; i<d; i++) s *= dd;
    ns = 1;
    for (i=0; i<d; i++) {
      s /= dd;
      ss = ns; ns *= dd;
      for (c=-depth; c<=depth; c++) {
	for (j=0; j<s; j++) {
	  for (k=0; k<ss; k++) coordinates[i+((c+depth)*ss+j*ns+k)*d] = c;
	}
      }
    }
  } else {
    return MPI_ERR_UNKNOWN;
  }

  return MPI_SUCCESS;
}

// stencil difference for two (almost) lexicographically ordered stencils

static int vectorequal(int d, int vec0[], int vec1[]) {
  int i;

  for (i=0; i<d; i++) if (vec0[i]!=vec1[i]) return 0;
  return 1;
}

int TUW_Cart_stencil_diff(MPI_Comm cart,
			  int n0, int coordinates0[],
			  int n1, int coordinates1[],
			  int *n, int coordinates[])
{
  nameattr *cartname;
  int flag;
  int i, i0, i1, j;
  int d;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;

  i0 = 0; i1 = 0; i = 0;
  while (i0<n0&&i1<n1) {
    if (!vectorequal(d,&coordinates0[i0*d],&coordinates1[i1*d])) {
      // copy out
      for (j=0; j<d; j++) coordinates[i*d+j] = coordinates0[i0*d+j];
      i0++; i++;
    } else {
      i0++; i1++;
    }
  }
  while (i0<n0) {
    // copy out
    for (j=0; j<d; j++) coordinates[i*d+j] = coordinates0[i0*d+j];
    i0++; i++;
  }
  *n = i;

  return MPI_SUCCESS;
}

int TUW_Cart_create_sub(MPI_Comm cart, int keepdimension[], MPI_Comm *subcart)
{
  nameattr *cartname;
  int flag;
  int i;
  int rank, sub;
  int d;
  
  MPI_Comm_get_attr(cart,namekey(),&cartname,&flag);
  if (!flag) return MPI_ERR_COMM;

  d = cartname->d;
  int coordinate[d];

  MPI_Comm_rank(cart,&rank);
  TUW_Cart_coordinates(cart,rank,coordinate);
  
  for (i=0; i<cartname->d; i++) if (keepdimension[i]) coordinate[i] = 0;
  TUW_Cart_rank(cart,coordinate,&sub);
  MPI_Comm_split(cart,sub,0,subcart);
  
  return MPI_SUCCESS;
}

// construct subgraph from dist-graph communicator
int TUW_Comm_subgraph(MPI_Comm distcomm, MPI_Comm subcomm,
		      MPI_Comm *subdistcomm)
{

  int commtype;
  
  MPI_Topo_test(distcomm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    MPI_Comm_dup(subcomm,subdistcomm);
  } else if (commtype==MPI_DIST_GRAPH) {
    MPI_Group distgroup, subgroup;
    int distsize, subsize;
    int i;
    
    int s, t;
    int w;
    MPI_Comm_group(distcomm,&distgroup);
    MPI_Comm_group(subcomm,&subgroup);
    MPI_Group_size(distgroup,&distsize);
    MPI_Group_size(subgroup,&subsize);

    MPI_Dist_graph_neighbors_count(distcomm,&s,&t,&w);

    int source[s]; // not on stack for large s and t
    int sourceweight[s];
    int target[t];
    int targetweight[t];
    MPI_Dist_graph_neighbors(distcomm,
			     s,source,sourceweight,
			     t,target,targetweight);

    int subsource[s];
    int subsourceweight[s];
    int ss;
    MPI_Group_translate_ranks(distgroup,s,source,subgroup,subsource);
    ss = 0;
    for (i=0; i<s; i++) {
      if (subsource[i]!=MPI_UNDEFINED) {
	subsource[ss] = subsource[i];
	assert(subsource[ss]<subsize);
	subsourceweight[ss] = subsourceweight[i];
	ss++;
      }
    }
    
    int subtarget[t];
    int subtargetweight[t];
    int tt;
    MPI_Group_translate_ranks(distgroup,t,target,subgroup,subtarget);
    tt = 0;
    for (i=0; i<t; i++) {
      if (subtarget[i]!=MPI_UNDEFINED) {
	subtarget[tt] = subtarget[i];
	assert(subtarget[tt]<subsize);
	subtargetweight[tt] = subtargetweight[i];
	tt++;
      }
    }

    if (w) {
      MPI_Dist_graph_create_adjacent(subcomm,ss,subsource,subsourceweight,
				     tt,subtarget,subtargetweight,
				     MPI_INFO_NULL,0,
				     subdistcomm);
    } else {
      MPI_Dist_graph_create_adjacent(subcomm,ss,subsource,MPI_UNWEIGHTED,
				     tt,subtarget,MPI_UNWEIGHTED,
				     MPI_INFO_NULL,0,
				     subdistcomm);
    }

    MPI_Group_free(&distgroup);
    MPI_Group_free(&subgroup);
  } else {
  }
  
  return MPI_SUCCESS;
}

// internal helper function:
// Testing Cartseianness of cartcomm created from named comm
// fully periodic grids for now (all neighborhoods same size, same stencil)
int tuw_isCartesian(MPI_Comm comm, MPI_Comm cart)
{
  int commtype;
  int flag;
  
  MPI_Topo_test(cart,&commtype);

  if (commtype==MPI_UNDEFINED) {
    flag = 0;
    goto decided;
  } else if (commtype==MPI_DIST_GRAPH) {
    int d;
    int rank, size;
    
    // Check whether the calling communicator has Cartesian naming.
    TUW_Cart_testall(comm,&flag,&d,&size);
    if (!flag) goto decided;
    assert(d>=1);

    int s,ss,t,tt;
    int w;

    // Process 0 broadcasts the number of its (source) neighbors.
    MPI_Dist_graph_neighbors_count(cart,&s,&t,&w);

    MPI_Comm_rank(cart,&rank);
    MPI_Comm_size(cart,&size);
    
    if (rank==0) {
      MPI_Bcast(&s,1,MPI_INT,0,cart);
      ss = s;
    } else {
      MPI_Bcast(&ss,1,MPI_INT,0,cart);
    }
    // All processes compare their own number of neighbors to that of process 0
    flag = (ss==s);
    MPI_Allreduce(MPI_IN_PLACE,&flag,1,MPI_INT,MPI_LAND,cart);
    if (!flag) goto decided;

    int coords[d];
    int rankcoords[d];
    int source[s];
    int target[t];
    int normal[t];

    TUW_Cart_coordinates(comm,rank,rankcoords);

    MPI_Dist_graph_neighbors(cart,
			     s,source,MPI_UNWEIGHTED,t,target,MPI_UNWEIGHTED);
    
    int i, j;
    for (i=0; i<t; i++) {
      TUW_Cart_coordinates(comm,target[i],coords);
      for (j=0; j<d; j++) coords[j] = coords[j]-rankcoords[j];
      TUW_Cart_rank(comm,coords,&normal[i]);
      assert(0<=normal[i]&&normal[i]<size); // valid rank
    }
    // bucket sort
    int counts[size];
    for (i=0; i<size; i++) counts[i] = 0;
    for (i=0; i<t; i++) counts[normal[i]]++;
    j = 0;
    for (i=0; i<size; i++) {
      while (counts[i]>0) {
	normal[j++] = i; counts[i]--;
      }
    }
    assert(j==t); // all in order

    if (rank==0) {
      MPI_Bcast(normal,t,MPI_INT,0,cart);
      flag = 1;
    } else {
      int normal0[t];
      MPI_Bcast(normal0,t,MPI_INT,0,cart);
      // check
      for (i=0; i<t; i++) if (normal[i]!=normal0[i]) break;
      flag = (i==t);
    }
    MPI_Allreduce(MPI_IN_PLACE,&flag,1,MPI_INT,MPI_LAND,cart);
    if (!flag) goto decided;
  }
  
 decided:
  return flag;
}

// Collectives

#define BARRIER_TAG        110
#define BCAST_TAG          111
#define GATHER_TAG         112
#define SCATTER_TAG        113
#define REDUCE_TAG         114
#define ALLREDUCE_TAG      115
#define REDUCE_SCATTER_TAG 116

int TUW_Barrier(MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Barrier(comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);
    
    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);

    MPI_Request request[s];

    // use some unspecified type for NULL message
    for (i=0; i<s; i++) {
      MPI_Irecv(NULL,0,MPI_BYTE,source[i],BARRIER_TAG,
		comm,&request[i]);
    }
    for (i=0; i<t; i++) {
      MPI_Send(NULL,0,MPI_BYTE,target[i],BARRIER_TAG,comm);
    }
    MPI_Waitall(s,request,MPI_STATUSES_IGNORE);
  }

  return MPI_SUCCESS;
}

int TUW_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
	      MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Bcast(buffer,count,datatype,root,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);

    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);

    MPI_Comm_rank(comm,&rank);
    if (rank==root) {
      for (i=0; i<t; i++) {
	MPI_Send(buffer,count,datatype,target[i],BCAST_TAG,comm);
      }
    } else {
      for (i=0; i<s; i++) {
	if (source[i]==root) {
	  MPI_Recv(buffer,count,datatype,root,BCAST_TAG,
		   comm, MPI_STATUS_IGNORE);
	}
      }
    }
  }
  
  return MPI_SUCCESS;
}

int TUW_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	       void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
	       MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Gather(sendbuf,sendcount,sendtype,
		 recvbuf,recvcount,recvtype,root,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);

    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);
    
    MPI_Comm_rank(comm,&rank);
    if (rank==root) {
      MPI_Aint lb, extent;

      MPI_Type_get_extent(recvtype,&lb,&extent);
      
      for (i=0; i<s; i++) {
	if (source[i]==root) {
	  MPI_Sendrecv(sendbuf,sendcount,sendtype,0,GATHER_TAG,
		       (char*)recvbuf+i*recvcount*extent,
		       recvcount,recvtype,0,GATHER_TAG,
		       MPI_COMM_SELF,MPI_STATUS_IGNORE);
	} else {
	  MPI_Recv((char*)recvbuf+i*recvcount*extent,recvcount,recvtype,
		   source[i],GATHER_TAG,comm,MPI_STATUS_IGNORE);
	}
      }
    } else {
      for (i=0; i<t; i++) {
	if (target[i]==root) {
	  MPI_Send(sendbuf,sendcount,sendtype,root,GATHER_TAG,comm);
	}
      }
    }
  }
  
  return MPI_SUCCESS;
}

int TUW_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		void *recvbuf, int recvcounts[], int recvdispls[],
		MPI_Datatype recvtype, int root,
		MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Gatherv(sendbuf,sendcount,sendtype,
		  recvbuf,recvcounts,recvdispls,recvtype,root,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);

    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);
    
    MPI_Comm_rank(comm,&rank);
    if (rank==root) {
      MPI_Aint lb, extent;

      MPI_Type_get_extent(recvtype,&lb,&extent);
      
      for (i=0; i<s; i++) {
	if (source[i]==root) {
	  MPI_Sendrecv(sendbuf,sendcount,sendtype,0,GATHER_TAG,
		       (char*)recvbuf+recvdispls[i]*extent,
		       recvcounts[i],recvtype,0,GATHER_TAG,
		       MPI_COMM_SELF,MPI_STATUS_IGNORE);
	} else {
	  MPI_Recv((char*)recvbuf+recvdispls[i]*extent,recvcounts[i],recvtype,
		   source[i],GATHER_TAG,comm,MPI_STATUS_IGNORE);
	}
      }
    } else {
      for (i=0; i<t; i++) {
	if (target[i]==root) {
	  MPI_Send(sendbuf,sendcount,sendtype,root,GATHER_TAG,comm);
	}
      }
    }
  }
  
  return MPI_SUCCESS;
}

int TUW_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
		MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Scatter(sendbuf,sendcount,sendtype,
		  recvbuf,recvcount,recvtype,root,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);

    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);
    
    MPI_Comm_rank(comm,&rank);
    if (rank==root) {
      MPI_Aint lb, extent;

      MPI_Type_get_extent(sendtype,&lb,&extent);
      
      for (i=0; i<t; i++) {
	if (target[i]==root) {
	  MPI_Sendrecv((char*)sendbuf+i*sendcount*extent,sendcount,sendtype,
		       0,GATHER_TAG,
		       recvbuf,recvcount,recvtype,0,GATHER_TAG,
		       MPI_COMM_SELF,MPI_STATUS_IGNORE);
	} else {
	  MPI_Send((char*)sendbuf+i*sendcount*extent,sendcount,sendtype,
		   target[i],GATHER_TAG,comm);
	}
      }
    } else {
      for (i=0; i<s; i++) {
	if (source[i]==root) {
	  MPI_Recv(recvbuf,recvcount,recvtype,
		   root,GATHER_TAG,comm,MPI_STATUS_IGNORE);
	}
      }
    }
  }

  return MPI_SUCCESS;
}

int TUW_Scatterv(void *sendbuf, int sendcounts[], int senddispls[],
		 MPI_Datatype sendtype,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
		 MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Scatterv(sendbuf,sendcounts,senddispls,sendtype,
		   recvbuf,recvcount,recvtype,root,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);

    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);
    
    MPI_Comm_rank(comm,&rank);
    if (rank==root) {
      MPI_Aint lb, extent;

      MPI_Type_get_extent(sendtype,&lb,&extent);
      
      for (i=0; i<t; i++) {
	if (target[i]==root) {
	  MPI_Sendrecv((char*)sendbuf+senddispls[i]*extent,
		       sendcounts[i],sendtype,0,GATHER_TAG,
		       recvbuf,recvcount,recvtype,0,GATHER_TAG,
		       MPI_COMM_SELF,MPI_STATUS_IGNORE);
	} else {
	  MPI_Send((char*)sendbuf+senddispls[i]*extent,sendcounts[i],sendtype,
		   target[i],GATHER_TAG,comm);
	}
      }
    } else {
      for (i=0; i<s; i++) {
	if (source[i]==root) {
	  MPI_Recv(recvbuf,recvcount,recvtype,
		   root,GATHER_TAG,comm,MPI_STATUS_IGNORE);
	}
      }
    }
  }

  return MPI_SUCCESS;
}

int TUW_Reduce(void *sendbuf,
	       void *recvbuf, int count, MPI_Datatype datatype,
	       MPI_Op op, int root, MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Reduce(sendbuf,recvbuf,count,datatype,op,root,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i, k;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);
    
    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);
    
    MPI_Comm_rank(comm,&rank);

    MPI_Request request[t];

    k = 0;
    for (i=0; i<t; i++) {
      if (target[i]==root) {
	MPI_Isend(sendbuf,count,datatype,root,REDUCE_TAG,comm,&request[k++]);
      }
    }

    if (rank==root&&s>0) {
      MPI_Aint lb, extent;
      void *tempbuf;
      
      MPI_Type_get_extent(datatype,&lb,&extent);

      MPI_Recv(recvbuf,count,datatype,source[0],REDUCE_TAG,
	       comm,MPI_STATUS_IGNORE);

      // alloc tempbuf
      tempbuf = (void*)malloc(count*extent);
      
      for (i=1; i<s; i++) {
	MPI_Recv(tempbuf,count,datatype,source[i],REDUCE_TAG,
		 comm,MPI_STATUS_IGNORE);
	MPI_Reduce_local(tempbuf,recvbuf,count,datatype,op);
      }

      free(tempbuf);
    }

    MPI_Waitall(k,request,MPI_STATUS_IGNORE);
  }

  return MPI_SUCCESS;
}

int TUW_Allreduce(void *sendbuf,
		  void *recvbuf, int count, MPI_Datatype datatype,
		  MPI_Op op, MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return
      MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    int rank;
    int s, t; // number of sources and targets
    int weighted;
    int i, k;
    
    MPI_Dist_graph_neighbors_count(comm,&s,&t,&weighted);
    
    int source[s];
    int target[t];
    
    MPI_Dist_graph_neighbors(comm,
			     s,source,MPI_UNWEIGHTED,
			     t,target,MPI_UNWEIGHTED);
    
    MPI_Comm_rank(comm,&rank);

    MPI_Request request[t];

    k = 0;
    for (i=0; i<t; i++) {
      MPI_Isend(sendbuf,count,datatype,target[i],ALLREDUCE_TAG,
		comm,&request[k++]);
    }

    if (s>0) {
      MPI_Aint lb, extent;
      void *tempbuf;
      
      MPI_Type_get_extent(datatype,&lb,&extent);

      MPI_Recv(recvbuf,count,datatype,source[0],ALLREDUCE_TAG,
	       comm,MPI_STATUS_IGNORE);

      // alloc tempbuf
      tempbuf = (void*)malloc(count*extent);
      
      for (i=1; i<s; i++) {
	MPI_Recv(tempbuf,count,datatype,source[i],ALLREDUCE_TAG,
		 comm,MPI_STATUS_IGNORE);
	MPI_Reduce_local(tempbuf,recvbuf,count,datatype,op);
      }

      free(tempbuf);
    }

    MPI_Waitall(k,request,MPI_STATUS_IGNORE);
  }
  
  return MPI_SUCCESS;
}

int TUW_Reduce_scatter_block(void *sendbuf,
			     void *recvbuf, int count, MPI_Datatype datatype,
			     MPI_Op op, MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Reduce_scatter_block(sendbuf,recvbuf,count,datatype,op,comm);
  } else {
    return MPI_ERR_UNKNOWN;
  }
}
  
int TUW_Reduce_scatter(void *sendbuf,
		       void *recvbuf, int counts[], MPI_Datatype datatype,
		       MPI_Op op, MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Reduce_scatter(sendbuf,recvbuf,counts,datatype,op,comm);
  } else {
    return MPI_ERR_UNKNOWN;
  }
}

int TUW_Scan(void *sendbuf,
	     void *recvbuf, int count, MPI_Datatype datatype,
	     MPI_Op op, MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Scan(sendbuf,recvbuf,count,datatype,op,comm);
  } else {
    return MPI_ERR_UNKNOWN;
  }
}

int TUW_Exscan(void *sendbuf,
	       void *recvbuf, int count, MPI_Datatype datatype,
	       MPI_Op op, MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Exscan(sendbuf,recvbuf,count,datatype,op,comm);
  } else {
    return MPI_ERR_UNKNOWN;
  }
}

int TUW_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		  void *recvbuf, int recvcount, MPI_Datatype recvtype,
		  MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Allgather(sendbuf,sendcount,sendtype,
			 recvbuf,recvcount,recvtype,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    return MPI_Neighbor_allgather(sendbuf,sendcount,sendtype,
				  recvbuf,recvcount,recvtype,comm);
  } else return MPI_ERR_UNKNOWN;
  
  return MPI_SUCCESS;
}

int TUW_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		   void *recvbuf, int recvcounts[], int recvdispls[],
		   MPI_Datatype recvtype,
		   MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Allgatherv(sendbuf,sendcount,sendtype,
			  recvbuf,recvcounts,recvdispls,recvtype,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    return MPI_Neighbor_allgatherv(sendbuf,sendcount,sendtype,
				   recvbuf,recvcounts,recvdispls,recvtype,comm);
  } else return MPI_ERR_UNKNOWN;
  
  return MPI_SUCCESS;
}

int TUW_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype,
		 MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Alltoall(sendbuf,sendcount,sendtype,
			recvbuf,recvcount,recvtype,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    return MPI_Neighbor_alltoall(sendbuf,sendcount,sendtype,
				 recvbuf,recvcount,recvtype,comm);
  } else return MPI_ERR_UNKNOWN;
  
  return MPI_SUCCESS;
}

int TUW_Alltoallv(void *sendbuf, int sendcounts[], int senddispls[],
		  MPI_Datatype sendtype,
		  void *recvbuf, int recvcounts[], int recvdispls[],
		  MPI_Datatype recvtype,
		  MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    return MPI_Alltoallv(sendbuf,sendcounts,senddispls,sendtype,
			 recvbuf,recvcounts,recvdispls,recvtype,comm);
  } else if (commtype==MPI_DIST_GRAPH) {
    return MPI_Neighbor_alltoallv(sendbuf,sendcounts,senddispls,sendtype,
				  recvbuf,recvcounts,recvdispls,recvtype,comm);
  } else return MPI_ERR_UNKNOWN;
  
  return MPI_SUCCESS;
}

int TUW_Alltoallw(void *sendbuf, int sendcounts[], MPI_Aint senddispls[],
		  MPI_Datatype sendtypes[],
		  void *recvbuf, int recvcounts[], MPI_Aint recvdispls[],
		  MPI_Datatype recvtypes[],
		  MPI_Comm comm)
{
  int commtype;
  
  MPI_Topo_test(comm,&commtype);

  if (commtype==MPI_UNDEFINED) {
    int *sdispls, *rdispls;
    int size;
    int i;
    
    MPI_Comm_size(comm,&size);

    // conversion to int displacements due to unfortunate MPI signature
    sdispls = (int*)malloc(size*sizeof(int));
    for (i=0; i<size; i++) sdispls[i] = (int)senddispls[i];
    rdispls = (int*)malloc(size*sizeof(int));
    for (i=0; i<size; i++) rdispls[i] = (int)recvdispls[i];
    return MPI_Alltoallw(sendbuf,sendcounts,sdispls,sendtypes,
			 recvbuf,recvcounts,rdispls,recvtypes,comm);

    free(sdispls);
    free(rdispls);
  } else if (commtype==MPI_DIST_GRAPH) {
    return MPI_Neighbor_alltoallw(sendbuf,sendcounts,senddispls,sendtypes,
				  recvbuf,recvcounts,recvdispls,recvtypes,comm);
  } else return MPI_ERR_UNKNOWN;
  
  return MPI_SUCCESS;
}
