.set	noreorder
.global	__setjmp
.global	_setjmp
.global	setjmp
.type	__setjmp,@function
.type	_setjmp,@function
.type	setjmp,@function
__setjmp:
_setjmp:
setjmp:
	sd	$ra, 0($4)
	sd	$sp, 8($4)
	sd	$gp, 16($4)
	sd	$16, 24($4)
	sd	$17, 32($4)
	sd	$18, 40($4)
	sd	$19, 48($4)
	sd	$20, 56($4)
	sd	$21, 64($4)
	sd	$22, 72($4)
	sd	$23, 80($4)
	sd	$30, 88($4)
#ifndef __mips_soft_float
	ldc1	$20, 96($4)
	ldc1	$22, 104($4)
	ldc1	$24, 112($4)
	ldc1	$26, 120($4)
	ldc1	$28, 128($4)
	ldc1	$30, 136($4)
#endif
	jr	$ra
	li	$2, 0
