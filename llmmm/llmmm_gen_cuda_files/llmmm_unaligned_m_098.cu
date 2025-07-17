#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_98_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<98> mm_instantiator;

public:
  UnalignedM_98_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_98_MMInstantiatorWrapper__;

}  // namespace LLMMM
