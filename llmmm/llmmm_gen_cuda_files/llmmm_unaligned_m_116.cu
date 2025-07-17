#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_116_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<116> mm_instantiator;

public:
  UnalignedM_116_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_116_MMInstantiatorWrapper__;

}  // namespace LLMMM
