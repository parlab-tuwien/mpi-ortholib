/* Jesper Larsson Traff, January 2021 */

#define TUW_ROWMAJOR 1
#define TUW_COLMAJOR 2

#define TUW_MANHATTAN_DISTANCE 1
#define TUW_CHEBYCHEV_DISTANCE 2

int TUW_Comm_base(MPI_Comm comm, MPI_Comm *basecomm);

int TUW_Cart_name(MPI_Comm comm,
		  int d, int dimorder, int order[], int periodic[], int *size);
int TUW_Cart_test(MPI_Comm cart, int *flag, int *d, int *size);
int TUW_Cart_testall(MPI_Comm cart, int *flag, int *d, int *size);
int TUW_Cart_get(MPI_Comm cart, int *dimorder,
		 int maxd, int order[], int periodic[]);

int TUW_Cart_rank(MPI_Comm cart, int coordinates[], int *rank);
int TUW_Cart_coordinates(MPI_Comm cart, int rank, int coordinates[]);

int TUW_Cart_relative_rank(MPI_Comm cart, int source, int relative[],
			   int *rank);
int TUW_Cart_relative_coordinates(MPI_Comm cart, int source, int dest,
				  int relative[]);
int TUW_Cart_relative_shift(MPI_Comm cart, int rank, int relative[],
			    int *inrank, int *outrank);
int TUW_Cart_allranks(MPI_Comm cart, int n, int coordinates[], int ranks[]);
int TUW_Cart_allranks_relative(MPI_Comm cart, int source,
			       int n, int reatives[], int ranks[]);

int TUW_Cart_neighbors_count(MPI_Comm cart, int rank,
			     int metric, int shadow, int depth, int *size);
int TUW_Cart_neighbors(MPI_Comm cart, int rank,
		       int metric, int shadow, int depth, int maxsize,
		       int neighbors[]);
int TUW_Cart_stencil_count(MPI_Comm cart, int metric, int depth, int *size);
int TUW_Cart_stencil(MPI_Comm cart, int metric, int depth,
		     int maxsize, int coordinates[]);
int TUW_Cart_stencil_diff(MPI_Comm cart,
			  int n0, int coordinates0[],
			  int n1, int coordinates1[],
			  int *n, int coordinates[]);

int TUW_Cart_create_sub(MPI_Comm cart, int dimension[], MPI_Comm *subcart);
int TUW_Comm_subgraph(MPI_Comm distcomm, MPI_Comm subcomm,
		      MPI_Comm *subdistcomm);

// "hidden" predicate
int tuw_isCartesian(MPI_Comm comm, MPI_Comm cart);

// The collectives

int TUW_Barrier(MPI_Comm comm);
int TUW_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
	      MPI_Comm comm);
int TUW_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
	       void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
	       MPI_Comm comm);
int TUW_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		void *recvbuf, int recvcounts[], int recvdispls[],
		MPI_Datatype recvtype, int root,
		MPI_Comm comm);
int TUW_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
		MPI_Comm comm);
int TUW_Scatterv(void *sendbuf, int sendcounts[], int senddispls[],
		 MPI_Datatype sendtype,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
		 MPI_Comm comm);
int TUW_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		  void *recvbuf, int recvcount, MPI_Datatype recvtype,
		  MPI_Comm comm);
int TUW_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		   void *recvbuf, int recvcounts[], int recvdispls[],
		   MPI_Datatype recvtype,
		   MPI_Comm comm);
int TUW_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype,
		 MPI_Comm comm);
int TUW_Alltoallv(void *sendbuf, int sendcounts[], int senddispls[],
		  MPI_Datatype sendtype,
		  void *recvbuf, int recvcounts[], int recvdispls[],
		  MPI_Datatype recvtype,
		  MPI_Comm comm);
int TUW_Alltoallw(void *sendbuf, int sendcounts[], MPI_Aint senddispls[],
		  MPI_Datatype sendtypes[],
		  void *recvbuf, int recvcounts[], MPI_Aint recvdispls[],
		  MPI_Datatype recvtypes[],
		  MPI_Comm comm);
int TUW_Reduce(void *sendbuf,
	       void *recvbuf, int count, MPI_Datatype datatype,
	       MPI_Op op, int root, MPI_Comm comm);
int TUW_Allreduce(void *sendbuf,
		  void *recvbuf, int count, MPI_Datatype datatype,
		  MPI_Op op, MPI_Comm comm);
int TUW_Reduce_scatter_block(void *sendbuf,
			     void *recvbuf, int count, MPI_Datatype datatype,
			     MPI_Op op, MPI_Comm comm);
int TUW_Reduce_scatter(void *sendbuf,
		       void *recvbuf, int counts[], MPI_Datatype datatype,
		       MPI_Op op, MPI_Comm comm);
