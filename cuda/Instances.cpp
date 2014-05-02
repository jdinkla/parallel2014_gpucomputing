/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#include "AddFunctor.h"
#include "IdFunctor.h"

template IdFunctor<int>;
template IdFunctor<float>;

template AddFunctor<int>;
template AddFunctor<float>;
