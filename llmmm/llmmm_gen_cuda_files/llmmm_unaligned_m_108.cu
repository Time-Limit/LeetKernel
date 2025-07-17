#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_108_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<108> mm_instantiator;

public:
  UnalignedM_108_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_108_MMInstantiatorWrapper__;

}  // namespace LLMMM
