/* basé sur on smallpt, a Path Tracer by Kevin Beason, 2008
 *  	http://www.kevinbeason.com/smallpt/ 
 *
 * Converti en C et modifié par Charles Bouillaguet, 2019
 *
 * Pour des détails sur le processus de rendu, lire :
 * 	https://docs.google.com/open?id=0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw
 */

#define _XOPEN_SOURCE
#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/stat.h>  /* pour mkdir    */ 
#include <unistd.h>    /* pour getuid   */
#include <sys/types.h> /* pour getpwuid */
#include <pwd.h>       /* pour getpwuid */
#include <mpi.h>



double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


enum Refl_t {DIFF, SPEC, REFR};   /* types de matériaux (DIFFuse, SPECular, REFRactive) */

struct Sphere { 
	double radius; 
	double position[3];
	double emission[3];     /* couleur émise (=source de lumière) */
	double color[3];        /* couleur de l'objet RGB (diffusion, refraction, ...) */
	enum Refl_t refl;       /* type de reflection */
	double max_reflexivity;
};

static const int KILL_DEPTH = 7;
static const int SPLIT_DEPTH = 4;

/* la scène est composée uniquement de spheres */
struct Sphere spheres[] = { 
// radius position,                         emission,     color,              material 
   {1e5,  { 1e5+1,  40.8,       81.6},      {},           {.75,  .25,  .25},  DIFF, -1}, // Left 
   {1e5,  {-1e5+99, 40.8,       81.6},      {},           {.25,  .25,  .75},  DIFF, -1}, // Right 
   {1e5,  {50,      40.8,       1e5},       {},           {.75,  .75,  .75},  DIFF, -1}, // Back 
   {1e5,  {50,      40.8,      -1e5 + 170}, {},           {},                 DIFF, -1}, // Front 
   {1e5,  {50,      1e5,        81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Bottom 
   {1e5,  {50,     -1e5 + 81.6, 81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, // Top 
   {16.5, {40,      16.5,       47},        {},           {.999, .999, .999}, SPEC, -1}, // Mirror 
   {16.5, {73,      46.5,       88},        {},           {.999, .999, .999}, REFR, -1}, // Glass 
   {10,   {15,      45,         112},       {},           {.999, .999, .999}, DIFF, -1}, // white ball
   {15,   {16,      16,         130},       {},           {.999, .999, 0},    REFR, -1}, // big yellow glass
   {7.5,  {40,      8,          120},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass middle
   {8.5,  {60,      9,          110},        {},           {.999, .999, 0   }, REFR, -1}, // small yellow glass right
   {10,   {80,      12,         92},        {},           {0, .999, 0},       DIFF, -1}, // green ball
   {600,  {50,      681.33,     81.6},      {12, 12, 12}, {},                 DIFF, -1},  // Light 
   {5,    {50,      75,         81.6},      {},           {0, .682, .999}, DIFF, -1}, // occlusion, mirror
}; 


/********** micro BLAS LEVEL-1 + quelques fonctions non-standard **************/
static inline void copy(const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] = x[i];
} 

static inline void zero(double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] = 0;
} 

static inline void axpy(double alpha, const double *x, double *y)
{
	for (int i = 0; i < 3; i++)
		y[i] += alpha * x[i];
} 

static inline void scal(double alpha, double *x)
{
	for (int i = 0; i < 3; i++)
		x[i] *= alpha;
} 

static inline double dot(const double *a, const double *b)
{ 
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} 

static inline double nrm2(const double *a)
{
	return sqrt(dot(a, a));
}

/********* fonction non-standard *************/
static inline void mul(const double *x, const double *y, double *z)
{
	for (int i = 0; i < 3; i++)
		z[i] = x[i] * y[i];
} 

static inline void normalize(double *x)
{
	scal(1 / nrm2(x), x);
}

/* produit vectoriel */
static inline void cross(const double *a, const double *b, double *c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

/****** tronque *************/
static inline void clamp(double *x)
{
	for (int i = 0; i < 3; i++) {
		if (x[i] < 0)
			x[i] = 0;
		if (x[i] > 1)
			x[i] = 1;
	}
} 

/******************************* calcul des intersections rayon / sphere *************************************/
   
// returns distance, 0 if nohit 
double sphere_intersect(const struct Sphere *s, const double *ray_origin, const double *ray_direction)
{ 
	double op[3];
	// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
	copy(s->position, op);
	axpy(-1, ray_origin, op);
	double eps = 1e-4;
	double b = dot(op, ray_direction);
	double discriminant = b * b - dot(op, op) + s->radius * s->radius; 
	if (discriminant < 0)
		return 0;   /* pas d'intersection */
	else 
		discriminant = sqrt(discriminant);
	/* détermine la plus petite solution positive (i.e. point d'intersection le plus proche, mais devant nous) */
	double t = b - discriminant;
	if (t > eps) {
		return t;
	} else {
		t = b + discriminant;
		if (t > eps)
			return t;
		else
			return 0;  /* cas bizarre, racine double, etc. */
	}
}

/* détermine si le rayon intersecte l'une des spere; si oui renvoie true et fixe t, id */
bool intersect(const double *ray_origin, const double *ray_direction, double *t, int *id)
{ 
	int n = sizeof(spheres) / sizeof(struct Sphere);
	double inf = 1e20; 
	*t = inf;
	for (int i = 0; i < n; i++) {
		double d = sphere_intersect(&spheres[i], ray_origin, ray_direction);
		if ((d > 0) && (d < *t)) {
			*t = d;
			*id = i;
		} 
	}
	return *t < inf;
} 

/* calcule (dans out) la lumiance reçue par la camera sur le rayon donné */
void radiance(const double *ray_origin, const double *ray_direction, int depth, unsigned short *PRNG_state, double *out)
{ 
	int id = 0;                             // id de la sphère intersectée par le rayon
	double t;                               // distance à l'intersection
	if (!intersect(ray_origin, ray_direction, &t, &id)) {
		zero(out);    // if miss, return black 
		return; 
	}
	const struct Sphere *obj = &spheres[id];
	
	/* point d'intersection du rayon et de la sphère */
	double x[3];
	copy(ray_origin, x);
	axpy(t, ray_direction, x);
	
	/* vecteur normal à la sphere, au point d'intersection */
	double n[3];  
	copy(x, n);
	axpy(-1, obj->position, n);
	normalize(n);
	
	/* vecteur normal, orienté dans le sens opposé au rayon 
	   (vers l'extérieur si le rayon entre, vers l'intérieur s'il sort) */
	double nl[3];
	copy(n, nl);
	if (dot(n, ray_direction) > 0)
		scal(-1, nl);
	
	/* couleur de la sphere */
	double f[3];
	copy(obj->color, f);
	double p = obj->max_reflexivity;

	/* processus aléatoire : au-delà d'une certaine profondeur,
	   décide aléatoirement d'arrêter la récusion. Plus l'objet est
	   clair, plus le processus a de chance de continuer. */
	depth++;
	if (depth > KILL_DEPTH) {
		if (erand48(PRNG_state) < p) {
			scal(1 / p, f); 
		} else {
			copy(obj->emission, out);
			return;
		}
	}

	/* Cas de la réflection DIFFuse (= non-brillante). 
	   On récupère la luminance en provenance de l'ensemble de l'univers. 
	   Pour cela : (processus de monte-carlo) on choisit une direction
	   aléatoire dans un certain cone, et on récupère la luminance en 
	   provenance de cette direction. */
	if (obj->refl == DIFF) {
		double r1 = 2 * M_PI * erand48(PRNG_state);  /* angle aléatoire */
		double r2 = erand48(PRNG_state);             /* distance au centre aléatoire */
		double r2s = sqrt(r2); 
		
		double w[3];   /* vecteur normal */
		copy(nl, w);
		
		double u[3];   /* u est orthogonal à w */
		double uw[3] = {0, 0, 0};
		if (fabs(w[0]) > .1)
			uw[1] = 1;
		else
			uw[0] = 1;
		cross(uw, w, u);
		normalize(u);
		
		double v[3];   /* v est orthogonal à u et w */
		cross(w, u, v);
		
		double d[3];   /* d est le vecteur incident aléatoire, selon la bonne distribution */
		zero(d);
		axpy(cos(r1) * r2s, u, d);
		axpy(sin(r1) * r2s, v, d);
		axpy(sqrt(1 - r2), w, d);
		normalize(d);
		
		/* calcule récursivement la luminance du rayon incident */
		double rec[3];
		radiance(x, d, depth, PRNG_state, rec);
		
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* dans les deux autres cas (réflection parfaite / refraction), on considère le rayon
	   réfléchi par la spère */

	double reflected_dir[3];
	copy(ray_direction, reflected_dir);
	axpy(-2 * dot(n, ray_direction), n, reflected_dir);

	/* cas de la reflection SPEculaire parfaire (==mirroir) */
	if (obj->refl == SPEC) { 
		double rec[3];
		/* calcule récursivement la luminance du rayon réflechi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		/* pondère par la couleur de la sphère, prend en compte l'emissivité */
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}

	/* cas des surfaces diélectriques (==verre). Combinaison de réflection et de réfraction. */
	bool into = dot(n, nl) > 0;      /* vient-il de l'extérieur ? */
	double nc = 1;                   /* indice de réfraction de l'air */
	double nt = 1.5;                 /* indice de réfraction du verre */
	double nnt = into ? (nc / nt) : (nt / nc);
	double ddn = dot(ray_direction, nl);
	
	/* si le rayon essaye de sortir de l'objet en verre avec un angle incident trop faible,
	   il rebondit entièrement */
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0) {
		double rec[3];
		/* calcule seulement le rayon réfléchi */
		radiance(x, reflected_dir, depth, PRNG_state, rec);
		mul(f, rec, out);
		axpy(1, obj->emission, out);
		return;
	}
	
	/* calcule la direction du rayon réfracté */
	double tdir[3];
	zero(tdir);
	axpy(nnt, ray_direction, tdir);
	axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), n, tdir);

	/* calcul de la réflectance (==fraction de la lumière réfléchie) */
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1 - (into ? -ddn : dot(tdir, n));
	double Re = R0 + (1 - R0) * c * c * c * c * c;   /* réflectance */
	double Tr = 1 - Re;                              /* transmittance */
	
	/* au-dela d'une certaine profondeur, on choisit aléatoirement si
	   on calcule le rayon réfléchi ou bien le rayon réfracté. En dessous du
	   seuil, on calcule les deux. */
	double rec[3];
	if (depth > SPLIT_DEPTH) {
		double P = .25 + .5 * Re;             /* probabilité de réflection */
		if (erand48(PRNG_state) < P) {
			radiance(x, reflected_dir, depth, PRNG_state, rec);
			double RP = Re / P;
			scal(RP, rec);
		} else {
			radiance(x, tdir, depth, PRNG_state, rec);
			double TP = Tr / (1 - P); 
			scal(TP, rec);
		}
	} else {
		double rec_re[3], rec_tr[3];
		radiance(x, reflected_dir, depth, PRNG_state, rec_re);
		radiance(x, tdir, depth, PRNG_state, rec_tr);
		zero(rec);
		axpy(Re, rec_re, rec);
		axpy(Tr, rec_tr, rec);
	}
	/* pondère, prend en compte la luminance */
	mul(f, rec, out);
	axpy(1, obj->emission, out);
	return;
}

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

int toInt(double x)
{
	return pow(x, 1 / 2.2) * 255 + .5;   /* gamma correction = 2.2 */
} 

int main(int argc, char **argv)
{ 
	/* Petit cas test (small, quick and dirty): */
	int w = 320;
	int h = 200;
	int samples = 200;

	/* Gros cas test (big, slow and pretty): */
	/* int w = 3840; */
	/* int h = 2160; */
	/* int samples = 5000;  */

	if (argc == 2) 
		samples = atoi(argv[1]) / 4;

	static const double CST = 0.5135;  /* ceci défini l'angle de vue */
	double camera_position[3] = {50, 52, 295.6};
	double camera_direction[3] = {0, -0.042612, -1};
	normalize(camera_direction);

	/* incréments pour passer d'un pixel à l'autre */
	double cx[3] = {w * CST / h, 0, 0};    
	double cy[3];
	cross(cx, camera_direction, cy);  /* cy est orthogonal à cx ET à la direction dans laquelle regarde la caméra */
	normalize(cy);
	scal(CST, cy);


	/* précalcule la norme infinie des couleurs */
	int n = sizeof(spheres) / sizeof(struct Sphere);
	for (int i = 0; i < n; i++) {
		double *f = spheres[i].color;
		if ((f[0] > f[1]) && (f[0] > f[2]))
			spheres[i].max_reflexivity = f[0]; 
		else {
			if (f[1] > f[2])
				spheres[i].max_reflexivity = f[1];
			else
				spheres[i].max_reflexivity = f[2]; 
		}
	}

	/* Début du parallélisme */

	 /* Chronometrage */
  	double debut, fin;
	/* Nb de processeurs et rang */
  	int rank, p, midwork,dem =-1, dem_send = dem, epsilon=1, end=0, I[2]; 

  	enum TYPE_MESSAGE{FINISH, ASK_FOR_WORK, SEND_WORK, SEND_SIZE, SEND_DATA};

 	 /* Initialisation de l'environnement MPI */
  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &p);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	MPI_Request reqsw, reqafw, reqwr, reqfwc;  // sw: send work, afw: ask for work, wr: work receive, fcw: finish work communications
  	int flagafw=0, flagwr=0, flagfwc=0;
  	MPI_Status statusafw, statuswr, statusfwc, status_size, status_data;

	/* Allocation de l'image */
	double *image = malloc(3 * w * h  * sizeof(*image));  //    /p
	if (image == NULL) {
		perror("Impossible d'allouer l'image");
		exit(1);
	}
	for (int i = 0; i < 3 * w * h ; i++)
	{
		image[i] = 0;
	}

	int useless, boucle_suivante = 0;

  	/* debut du chronometrage */
  	debut = my_gettimeofday();

  	int imin = (p-1-rank)*h/p;
  	int imax = (p-rank)*h/p;

  	MPI_Irecv(&dem, 1, MPI_INT, (rank-1)%p, ASK_FOR_WORK, MPI_COMM_WORLD, &reqafw);	 		//On se prépare à recevoir une demande de travail
  	MPI_Irecv(&useless, 1, MPI_INT, MPI_ANY_SOURCE, FINISH, MPI_COMM_WORLD, &reqfwc);		 //On se prépare à la fin des communications de demandes de travail
	
	// On garde en mémoire les taches effectues
	int Imin_memory[1024];
	int Imax_memory[1024];
	int nb_memory = 0;

  	printf("Processus %02d: (imin = %03d, imax = %03d) begin \n",rank, imin, imax-1);
	printf("\n");

  	while(!end && !flagfwc) {		// Tant que je n'ai pas fini en premier ou que personne n'a fini (flag3 qui nous prévient si qq1 d'autre a fini avant nous)

  			//printf("processus %d:  imin = %03d,\timax = %03d \n",rank, imin, imax-1);
  			for (int i = imin; i < imax; i++) {		//Début de la boucle de calcul


				/// ------------------------------------------------------------------------------------- ///
				/// ---------------------- Traitement des demandes de travail --------------------------- ///
				/// ------------------------------------------------------------------------------------- ///

  				MPI_Test(&reqfwc, &flagfwc, &statusfwc);	// On a encore besoin de traiter les demandes de travail ?
  				if(!flagfwc) {	//Oui

  					MPI_Test(&reqafw, &flagafw, &statusafw);	
  					if(flagafw) {							// traitement de la demande de travail

  						if((imax-i)>epsilon) {			//A t-on du travail à donner ? OUI
  							midwork = i+(imax-i)/2;		// Moitié du travail 
							if (i != imin)			// On note notre travail réalisé
							{
								Imin_memory[nb_memory] = imin;
								Imax_memory[nb_memory] = i - 1;
								nb_memory ++;
								printf("Processus %02d: (imin = %03d, imax = %03d) DONE \n\n", rank, imin, i-1);
							}

  							int I[] = {midwork,imax};	// Contenu du travail qu'on donne
  							MPI_Send(I, 2, MPI_INT, dem, SEND_WORK, MPI_COMM_WORLD); // On envoie le travail

							//printf("Processus %20d, en cours de travail a donné le travail (imin = %30d, imax = %30d) au processus %02d\n", rank, midwork, imax, dem);
  							imax = midwork;			// On met à jours notre nouvelle tache
  							imin = i;

							MPI_Irecv(&dem, 1, MPI_INT, (rank-1)%p, ASK_FOR_WORK, MPI_COMM_WORLD, &reqafw);		// On se remet en attente d'une demande de travail
							boucle_suivante = 1; // On va break et passer à la boucle de calcul suivante
							break;
  						}
  						else {							// Non, on transmet au collègue suivant
							dem_send = dem;
  							MPI_Isend(&dem_send, 1, MPI_INT, (rank+1)%p, ASK_FOR_WORK, MPI_COMM_WORLD, &reqsw);	
							MPI_Irecv(&dem, 1, MPI_INT, (rank-1)%p, ASK_FOR_WORK, MPI_COMM_WORLD, &reqafw);		// On se remet en attente d'une demande de travail

  							//printf("Le processus %20d, en cours de travail a transmis la demande de travail de %03d au processus %02d\n", rank, dem, (rank+1)%p);
						}
  					}
  				}

				/// ---------------------------------------------------------------------------- ///
				/// ---------------------- Boucle de calcul ------------------------------------ ///
				/// ---------------------------------------------------------------------------- ///

 			unsigned short PRNG_state[3] = {0, 0, i*i*i};
			for (unsigned short j = 0; j < w; j++) {

				/* calcule la luminance d'un pixel, avec sur-échantillonnage 2x2 */
				double pixel_radiance[3] = {0, 0, 0};
				for (int sub_i = 0; sub_i < 2; sub_i++) {
					for (int sub_j = 0; sub_j < 2; sub_j++) {
						double subpixel_radiance[3] = {0, 0, 0};
						/* simulation de monte-carlo : on effectue plein de lancers de rayons et on moyenne */
						for (int s = 0; s < samples; s++) { 
							/* tire un rayon aléatoire dans une zone de la caméra qui correspond à peu près au pixel à calculer */
							double r1 = 2 * erand48(PRNG_state);
							double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
							double r2 = 2 * erand48(PRNG_state);
							double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
							double ray_direction[3];
							copy(camera_direction, ray_direction);
							axpy(((sub_i + .5 + dy) / 2 + i) / h - .5, cy, ray_direction);
							axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);
							normalize(ray_direction);

							double ray_origin[3];
							copy(camera_position, ray_origin);
							axpy(140, ray_direction, ray_origin);
						
							/* estime la lumiance qui arrive sur la caméra par ce rayon */
							double sample_radiance[3];
							radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
							/* fait la moyenne sur tous les rayons */
							axpy(1. / samples, sample_radiance, subpixel_radiance);
						}
						clamp(subpixel_radiance);
						/* fait la moyenne sur les 4 sous-pixels */
						axpy(0.25, subpixel_radiance, pixel_radiance);
					}
				}
				//copy(pixel_radiance, image + 3 * (((p-rank)*h/p - 1 - i) * w + j)); // <-- retournement vertical
				copy(pixel_radiance, image + 3 * ((h - 1 - i) * w + j)); // <-- retournement vertical

			}// Fin d'un ligne de calcul		
		}

		if (boucle_suivante)	// Si nous a demandé du travail, on ne va pas demander du travail
		{
			boucle_suivante = 0;
			continue;
		}
		
		// On note notre travail realisé
		Imin_memory[nb_memory] = imin;
		Imax_memory[nb_memory] = imax-1;
		nb_memory++;
		printf("Processus %02d: (imin = %03d, imax = %03d) DONE \n\n", rank, imin, imax-1);

	
		/// ---------------------------------------------------------------------------- ///
		/// ---------------------- On va demander du travail --------------------------- ///
		/// ---------------------------------------------------------------------------- ///

		MPI_Test(&reqfwc, &flagfwc, &statusfwc);	// Fin des communications de demande de travail ?
		if(!flagfwc) {		// Non, on demande du travail
			MPI_Send(&rank, 1, MPI_INT, (rank+1)%p, ASK_FOR_WORK, MPI_COMM_WORLD);		// Envoi de la demande
			// printf("Le processus %d a envoyé une demande de travail au processus %d\n", rank, (rank+1)%p );
			MPI_Irecv(I, 2, MPI_INT, MPI_ANY_SOURCE, SEND_WORK, MPI_COMM_WORLD, &reqwr);		// On se prépare à recevoir du travail

			do{											// Traitement des messages reçus
				MPI_Test(&reqfwc, &flagfwc, &statusfwc);	// Si quelqu'un nous annonce la fin des communications de travail
				if(flagfwc) break;
				MPI_Test(&reqafw, &flagafw, &statusafw);		// Sinon on continue de traiter les demandes de travail
				if(flagafw)	//Réception d'une demande de travail
					{
						if (dem == rank)	// C'est la notre, fin des comm de travail
						{
							end = 1;
						}
						else // On transmet la demande du précédent processus au suivant
						{
							//printf("Le processus %d a transmis la demande de travail de %d au processus %d\n", rank, dem, (rank+1)%p);
							dem_send = dem;
  							MPI_Isend(&dem_send, 1, MPI_INT, (rank+1)%p, ASK_FOR_WORK, MPI_COMM_WORLD, &reqsw);
						}
  						MPI_Irecv(&dem, 1, MPI_INT, (rank-1)%p, ASK_FOR_WORK, MPI_COMM_WORLD, &reqafw); // On se remet en attente d'une demande de travail
					}

				MPI_Test(&reqwr, &flagwr, &statuswr);  // A t-on reçu du travail ?
			}
			while((flagwr == 0) && (end == 0)); 	// Tant que nous n'avons pas reçu de travail ou que nous n'avons pas été le premier à finir(reception de notre propre demande)

			MPI_Test(&reqfwc, &flagfwc, &statusfwc);	// Si quelqu'un nous annonce la fin des communications de travail
			if (flagwr && !flagfwc)  // On a reçu du travail --> Traitement
			{
			//	printf("Le processus %d a recu les taches (imin = %d, imax = %d)\n", rank, I[0], I[1]);
				imin=I[0];
				imax=I[1];
			}
			if (end && !flagfwc)   // On est le processus qui constate la fin en premier
			{
				printf("==================================================================================\n");
				printf("Le processus %d annonce à chaque processus qu'il est inutile de demander du travail\n", rank);
				printf("===================================================================================\n");
				
				for(int i = 0; i < p; i++)	// On annonce à chaque processus qu'il faut arrêter de demander du travail, il n'y en a plus !
				{
					MPI_Cancel(&reqfwc);
					if( i != rank )
					{
						MPI_Isend(&useless, 1, MPI_INT, i, FINISH, MPI_COMM_WORLD, &reqfwc);
					}
				}
				
			}
		}
	}


	fprintf( stdout, "\n");
	
	MPI_Cancel(&reqafw); // Plus besoin de savoir si quelqu'un nous a demandé une tache
	
  	/* fin du chronometrage */
 	fin = my_gettimeofday();
	//  fprintf( stdout, "Temps total de calcul pour le processus %d : %g sec end = %d flagfwc = %d\n", rank, fin - debut, end, flagfwc);
  	fprintf( stdout, "Temps total de calcul pour le processus %d : %g sec\n", rank, fin - debut);

	if (rank == 0)
	{
		debut = my_gettimeofday();
	}
	/*
	for(int i = 0; i < cursor ; i++)
	{
		fprintf(stdout, "Processus %d: Lignes %d - %d\n", rank, Imin[i], Imax[i]);
	}
	*/


	/// ---------------------------------------------------------------------------- ///
	/// ------------------------------ GATHER DES DONNES --------------------------- ///
	/// ---------------------------------------------------------------------------- ///

	// On va obtenir le nombre de tâche realise par les processus
	int nb_memory_sum = 0;
	MPI_Reduce( &nb_memory, &nb_memory_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	nb_memory_sum -= nb_memory;

	int I2[2]; // Contient les tâches
	if ( rank == 0 )
	{
		for (int i = 0; i < nb_memory_sum; i++) // On va recuperer toutes les taches, on somme sur le nombre de tache realise par les processus
		{
			MPI_Recv(I2, 2, MPI_INT, MPI_ANY_SOURCE, SEND_SIZE, MPI_COMM_WORLD, &status_size); // On recoit la tache i
			int source = status_size.MPI_SOURCE;							// On garde en memoire la source
			MPI_Recv(image + 3 * (h-1-I2[1])*w, 3*(I2[1] - I2[0]) * w + 3*w, MPI_DOUBLE, source, SEND_DATA, MPI_COMM_WORLD, &status_data); // On recoit les donnes correspondants à la tache i
		}	
	}
    	else
	{
		for (int i = 0; i < nb_memory; i++) // On va envoyer au processus 0 nos numeros de tache ainsi que les donnees correspondantes
		{
			I2[0] = Imin_memory[i] ;
			I2[1] = Imax_memory[i] ;
			MPI_Send(I2, 2, MPI_INT, 0, SEND_SIZE, MPI_COMM_WORLD); // On envoie les tâches
			MPI_Send(image + 3 * (h-1-I2[1])*w, 3* (I2[1] - I2[0] ) * w + 3*w , MPI_DOUBLE, 0 , SEND_DATA, MPI_COMM_WORLD); // On envoie les donnes correspondant à la tache
		}
	}


	/// ---------------------------------------------------------------------------- ///
	/// ------------------------------ ECRITURE DES DONNES ------------------------- ///
	/// ---------------------------------------------------------------------------- ///

        if( rank == 0 )
	{
        // stocke l'image dans un fichier au format NetPbm 
                {
                        struct passwd *pass; 
                        char nom_sortie[100] = "";
                        char nom_rep[30] = "";

                        pass = getpwuid(getuid()); 
                        sprintf(nom_rep, "/tmp/%s", pass->pw_name);
                        mkdir(nom_rep, S_IRWXU);
                        sprintf(nom_sortie, "%s/image.ppm", nom_rep);
              		printf("\nLancer l'image: display %s\n", nom_sortie);
			 
                        FILE *f = fopen(nom_sortie, "w");
                        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
                        for (int i = 0; i < w * h; i++) 
                                fprintf(f,"%d %d %d ", toInt(image[3 * i]), toInt(image[3 * i + 1]), toInt(image[3 * i + 2])); 
                        fclose(f); 
			/* fin du chronometrage */
			fin = my_gettimeofday();
			fprintf( stdout, "Temps pour rassembler et ecrire les donnes : %g sec\n", fin - debut);
                }
        }    
	free(image);

	MPI_Finalize();
}
