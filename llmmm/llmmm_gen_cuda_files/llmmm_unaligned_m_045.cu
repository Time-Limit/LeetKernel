#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_45_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<45> mm_instantiator;

public:
  UnalignedM_45_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_45_MMInstantiatorWrapper__;

}  // namespace LLMMM
