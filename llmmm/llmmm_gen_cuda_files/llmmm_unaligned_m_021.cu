#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_21_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<21> mm_instantiator;

public:
  UnalignedM_21_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_21_MMInstantiatorWrapper__;

}  // namespace LLMMM
