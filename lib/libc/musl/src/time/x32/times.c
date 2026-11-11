#include <sys/times.h>
#include "syscall.h"

clock_t times(struct tms *tms)
{
	long long ktms[4];
	clock_t r = __syscall(SYS_times, ktms);
	if (tms) {
		tms->tms_utime = ktms[0];
		tms->tms_stime = ktms[1];
		tms->tms_cutime = ktms[2];
		tms->tms_cstime = ktms[3];
	}
	return r;
}
